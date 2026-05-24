from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import gc
import threading
import numpy as np
import jack
from clock_generator import ClockGenerator

import wx
import queue

from data import Beat, GreedyBeat
from beatdetector import BeatDetector


CHUNKSIZE = 1*1024 # 8192

class JackHandler:
	client: jack.Client
	event: threading.Event
	ringbuf_in: jack.RingBuffer
	ringbuf_beats: jack.RingBuffer
	last_beatupdate_frames: int | None # [audio frames]
	last_beatupdate_tpb: int | None    # [audio frames]
	click_mask: np.ndarray
	clicks: list[int]
	n_clicks: int
	SINE_PERIOD_FRAMES: int
	sine: np.ndarray
	MIDI_CLOCK: list[int]
	midi_clock_generator: ClockGenerator

	def __init__(self, client: jack.Client):
		self.client = client
		self.event = threading.Event()
		self.ringbuf_in = jack.RingBuffer(max(2**16, 4*CHUNKSIZE)*4)
		self.ringbuf_beats = jack.RingBuffer(256)
		self.last_beatupdate_frames = None
		self.last_beatupdate_tpb = None
		self.click_mask = np.zeros(1024*32)
		self.clicks = [0]*16
		self.n_clicks = 0
		self.SINE_PERIOD_FRAMES = int(48000//880)
		self.sine = np.sin(np.arange(0, 1024*32) /self.SINE_PERIOD_FRAMES*2*3.141592654)
		self.MIDI_CLOCK=[0xF8]
		self.midi_clock_generator = ClockGenerator(24, min_delta = 10)

	def register_jack_callbacks(self) -> None:
		self.client.set_process_callback(self.process)
		self.client.set_shutdown_callback(self.on_shutdown)
		self.client.set_xrun_callback(self.on_xrun)
		self.client.set_samplerate_callback(self.on_samplerate)
		self.client.set_blocksize_callback(self.on_blocksize)

	def on_shutdown(self, status, reason): # type: ignore[no-untyped-def]
		print(f"shutting down. {reason}")
	
	def on_xrun(self, usecs): # type: ignore[no-untyped-def]
		print(f"XRUN {usecs}us")

	def on_samplerate(self, samplerate): # type: ignore[no-untyped-def]
		print(f"SAMPLE RATE CHANGE {samplerate}")

	def on_blocksize(self, blocksize): # type: ignore[no-untyped-def]
		print(f"BLOCKSIZE CHANGE {blocksize}")

	def read_input(self, chunksize: int) -> tuple[int, bytes]:
		assert chunksize > 0

		data = bytes()
		t0 = None
		while len(data) // 4 < chunksize:
			time, one_audio = self._read_one_input_segment()
			if t0 is None:
				t0 = time
			assert len(data) % 4 == 0
			if time != t0 + len(data)//4:
				print(f"ERROR: lost {time - (t0 + len(data)//4)} samples; time = {time}, t0 + len(data)/4 = {t0} + {len(data)//4} = {t0+len(data)//4}")
			#assert time == t0 + len(data) // 4
			data += one_audio

		assert t0 is not None, "t0 cannot be none because chunksize>0 means that the loop above has set t0 at least once"
		return t0, data

	def _read_one_input_segment(self) -> tuple[int, Any]:
		self.event.clear()

		while self.ringbuf_in.read_space < 8:
			self.event.wait()
			self.event.clear()

		frames_bytes = self.ringbuf_in.read(8)
		assert len(frames_bytes) == 8
		frames = int.from_bytes(frames_bytes, 'little')

		while self.ringbuf_in.read_space < 8 + 4*frames:
			self.event.wait()
			self.event.clear()

		time_bytes = self.ringbuf_in.read(8)
		assert len(time_bytes) == 8
		time = int.from_bytes(time_bytes, 'little')
		
		audio = self.ringbuf_in.read(4*frames)
		assert len(audio) == 4*frames

		return time, audio

	# time: time of the next beat [frames]
	# tpb: time between beats [frames]
	def update_beats(self, time: int, tpb: int) -> None:
		if self.ringbuf_beats.write_space >= 16:
			self.ringbuf_beats.write(int.to_bytes(time, 8, 'little'))
			self.ringbuf_beats.write(int.to_bytes(tpb, 8, 'little'))
		else:
			print("not enough space in ringbuf_beats")

	def process(self, frames: int) -> None:
		if frames == 0: return

		t0: int = self.client.last_frame_time

		midi_outport: jack.OwnMidiPort = self.client.midi_outports[0] # type: ignore
		midi_outport.clear_buffer()

		inbuf = self.client.inports[0].get_buffer() # type: ignore

		assert len(inbuf) == 4*frames
		n = self.ringbuf_in.write(int.to_bytes(frames, 8, 'little'))
		assert n == 8
		n = self.ringbuf_in.write(int.to_bytes(t0, 8, 'little'))
		assert n == 8
		n = self.ringbuf_in.write(inbuf)
		assert n == frames * 4
		self.event.set()

		if self.ringbuf_beats.read_space >= 16:
			# skip all updates but the latest
			self.ringbuf_beats.read_advance((self.ringbuf_beats.read_space // 16 - 1) * 16)
			# read the latest update
			buf = self.ringbuf_beats.read(16)
			last_beatupdate_frames = int.from_bytes(buf[0:8], 'little')
			last_beatupdate_tpb = int.from_bytes(buf[8:16], 'little')
			self.last_beatupdate_frames = last_beatupdate_frames
			self.last_beatupdate_tpb = last_beatupdate_tpb
			assert last_beatupdate_frames <= t0
			
			self.midi_clock_generator.update_beats(last_beatupdate_frames, last_beatupdate_tpb)

		CLICK_FRAMES=int(0.06 * self.client.samplerate)
		self.click_mask[0:frames] = 0
		if self.last_beatupdate_frames is not None:
			assert self.last_beatupdate_tpb is not None, "if last_beatupdate_frames is not None, then _tpb is neither"
			first_relevant_beat_index = (t0 - self.last_beatupdate_frames - CLICK_FRAMES + self.last_beatupdate_tpb-1) // self.last_beatupdate_tpb
			first_irrelevant_beat_index = (t0 + frames - self.last_beatupdate_frames + self.last_beatupdate_tpb-1) // self.last_beatupdate_tpb
			for i in range(first_relevant_beat_index, first_irrelevant_beat_index):
				start = self.last_beatupdate_frames + i*self.last_beatupdate_tpb - t0
				end = start + CLICK_FRAMES
				self.click_mask[max(0, start) : min(end, frames)] += 0.5

			self.midi_clock_generator.get_ticks_cb(t0, t0+frames, lambda time, _ : midi_outport.write_midi_event(int(time-t0), self.MIDI_CLOCK))
		
		sin_offset = t0 % self.SINE_PERIOD_FRAMES
		# FIXME this should be get_buffer
		self.client.outports[0].get_array()[:] = self.sine[sin_offset : sin_offset+frames] * self.click_mask[:frames] # type: ignore

@dataclass
class Tap:
	timestamp_samples: int
	samples_per_beat: int|None

# Call .tap(...) with various timestamps. Returns the time-per-beat,
# or None on the first tap in a sequence.
# Timestamps must be in int (to avoid float precision issues)
class TempoTapper:
	tap_history: list[int] # list of timepoints [audio frames]
	timeout: int

	def __init__(self, timeout: float):
		self.tap_history = []
		self.timeout = timeout

	def tap(self, now: int) -> int|None:
		if len(self.tap_history) > 0 and self.tap_history[-1] < now - self.timeout:
			print(f"clearing tap history ({self.tap_history[-1]} too old for {now})")
			self.tap_history = []

		self.tap_history.append(now)

		if len(self.tap_history) >= 2:
			return (self.tap_history[-1] - self.tap_history[0]) // (len(self.tap_history)-1)
		else:
			return None

class MyFrame(wx.Frame): #type: ignore[misc]
	tap_btn: wx.Button
	jackclient: jack.Client
	tempo_tapper: TempoTapper
	taps: queue.Queue[Tap]
	labels: list[wx.StaticText]
	tempolabel: wx.StaticText
	infolabel: wx.StaticText

	def __init__(self, jackclient: jack.Client):
		wx.Frame.__init__(self, None, wx.ID_ANY, "beatdetect")

		self.tap_btn = wx.Button(self, label="tap")
		self.tap_btn.SetOwnBackgroundColour(wx.BLUE)
		self.tap_btn.Bind(wx.EVT_LEFT_DOWN,self.on_tap)
		self.jackclient = jackclient
		self.tempo_tapper = TempoTapper(1.5*jackclient.samplerate)
		self.taps = queue.Queue(maxsize=100)

		self.labels = [wx.StaticText(self, label=f'tbd #{i}') for i in range(4)]
		self.tempolabel = wx.StaticText(self, label='tempo')
		self.infolabel = wx.StaticText(self, label='info')

		hbox = wx.BoxSizer(wx.HORIZONTAL)
		vbox1 = wx.BoxSizer(wx.VERTICAL)
		vbox2 = wx.BoxSizer(wx.VERTICAL)
		hbox.Add(vbox1, 5)
		hbox.Add(vbox2, 2)
		vbox1.Add(self.tap_btn)
		vbox1.Add(self.tempolabel)
		vbox1.Add(self.infolabel)
		for l in self.labels:
			vbox2.Add(l, 1)

		self.SetSizer(hbox)


	def on_tap(self, ev: wx.Event) -> None:
		ev.Skip() # because documentation on EVT_LEFT_DOWN says so
		now = self.jackclient.frame_time - self.jackclient.blocksize
		samples_per_beat: int|None = self.tempo_tapper.tap(now)
		self.taps.put(Tap(now, samples_per_beat))

	# schedules the button to flash at "when" [global audio frame time]
	def flash(self, when: int) -> None:
		delay_frames = when - self.jackclient.frame_time
		delay_millis = int(delay_frames / self.jackclient.samplerate * 1000)
		wx.CallLater(delay_millis, lambda : self.tap_btn.SetOwnBackgroundColour(wx.RED))
		wx.CallLater(delay_millis+60, lambda : self.tap_btn.SetOwnBackgroundColour(wx.BLUE))

	# sets the debug label texts
	def set_texts(self, texts: list[str], start:int=0) -> None:
		def doit() -> None:
			for label, text in zip(self.labels[start:], texts):
				label.SetLabel(text)
		wx.CallAfter(doit)

	# sets the BPM label
	def set_bpm(self, bpm: float) -> None:
		wx.CallAfter(lambda : self.tempolabel.SetLabel(f"{bpm:5.1f} bpm"))
	
	# sets the info label
	def set_info(self, info: str) -> None:
		wx.CallAfter(lambda : self.infolabel.SetLabel(info))
				
def run_live(timestep: float, force_bpm: float|None = None, enable_gui: bool = True) -> None:
	client = jack.Client('beatdetect')
	audio_in = client.inports.register('audio_in')
	click_out = client.outports.register('click_out')
	midiclock_out = client.midi_outports.register('midiclock_out')

	total_samples = 0
	last_greedy_beat = 0.0

	jackhandler = JackHandler(client)
	jackhandler.register_jack_callbacks()

	samplerate = client.samplerate
	bd = BeatDetector(samplerate, force_bpm = force_bpm, timestep_desired_ms = timestep)


	if enable_gui:
		app = wx.App(False)
		window = MyFrame(client)
		window.Show(True)
		threading.Thread(target = lambda : app.MainLoop()).start()

	last_our_frametime_to_jack: int = 0
	with client:
		print("hi")
		our_frametime = 0
		last_beatupdate_frames = 0
		last_beatupdate_tpb = 48000
		while True:
			jack_frametime, data_bytes = jackhandler.read_input(CHUNKSIZE)
			#print(f"got len(data_bytes) bytes / {len(data_bytes)//4} samples at {jack_frametime} vs {our_frametime}")
			assert len(data_bytes) >= CHUNKSIZE*4
			data = np.frombuffer(data_bytes, dtype=np.float32)
			frames = len(data)

			rms = np.sqrt(np.mean(np.square(data)))
			if len(bd.trackers) == 0 and rms > 1e-3 and not bd.is_tempo_estimation_pending():
				print("no trackers and non-silence detected, requesting tempo estimation")
				bd.request_tempo_estimation()

			# our audio buffer starts at jack_frametime

			our_frametime_to_jack = jack_frametime - our_frametime
			our_frametime += frames

			if our_frametime_to_jack != last_our_frametime_to_jack:
				print(f"Jump in frametime detected by {our_frametime_to_jack-last_our_frametime_to_jack} samples = {(our_frametime_to_jack-last_our_frametime_to_jack)/client.samplerate:.3f} sec")
				last_our_frametime_to_jack = our_frametime_to_jack

			if enable_gui:
				last_tap = None
				try:
					while True:
						last_tap = window.taps.get_nowait()
				except queue.Empty:
					pass

			if last_tap is not None:
				tap_ourframes = last_tap.timestamp_samples - our_frametime_to_jack

				new_tpb = None if last_tap.samples_per_beat is None else (bd.deltatimestep_from_deltasample(last_tap.samples_per_beat))

				bd.resync(bd.timestep_from_sample(tap_ourframes), new_tpb)
				if new_tpb is None:
					print("sync!")
				else:
					print("sync! %3.1f bpm" % (client.samplerate / last_tap.samples_per_beat * 60))

			#print(f"t = {total_samples/samplerate:5.1f}, got {frames} frames, data len is {len(data)}")

			bd.process(data)
			# FIXME this drops beats if there are more than one beat in the time frame.
			tpb = None
			if len(bd.greedy_beats) > 0:
				tpb = bd.greedy_beats[-1].tracker_time_per_beat
				if bd.greedy_beats[-1].location != last_greedy_beat:
					last_greedy_beat = bd.greedy_beats[-1].location

					beat_samples = int(bd.greedy_beats[-1].location * bd.timestep_real * samplerate)
					samples_per_beat = int(bd.greedy_beats[-1].tracker_time_per_beat * bd.timestep_real * samplerate)

					jackhandler.update_beats(beat_samples + our_frametime_to_jack, samples_per_beat)

					last_beatupdate_frames = beat_samples + our_frametime_to_jack
					last_beatupdate_tpb = samples_per_beat

			t0 = client.frame_time
			first_relevant_beat_index = (t0 - last_beatupdate_frames + last_beatupdate_tpb-1) // last_beatupdate_tpb
			first_irrelevant_beat_index = (t0 + max(CHUNKSIZE, frames) - last_beatupdate_frames + last_beatupdate_tpb-1) // last_beatupdate_tpb
			#print(first_relevant_beat_index)

			if enable_gui:
				for i in range(first_relevant_beat_index, first_irrelevant_beat_index):
					beat_t = last_beatupdate_frames + i*last_beatupdate_tpb
					window.flash(beat_t)

				window.set_texts(['%4.1f%% ( * %3.0f%%)' % (t.confidence*100, t.greedy_continuity*100) for t in bd.trackers[0:4]])
				bpm = 0 if tpb is None else (60 / (tpb * bd.timestep_real))
				window.set_bpm(bpm)
				#window.set_info(f"predicting {first_relevant_beat_index} .. {first_irrelevant_beat_index} beats ahead")
				#window.set_info(f"{gc.get_count()}")
				window.set_info(f"{len(bd.greedy_beats)} / {len(bd.trackers[0].beats) if len(bd.trackers)>0 else 0}")
				#print(gc.get_stats())

