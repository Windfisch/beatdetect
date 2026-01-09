from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import soundfile as sf
import gc
import threading
import math
import sys
import numpy as np
import scipy.signal as ss
import scipy.ndimage as sn
import scipy.sparse as sparse
from numpy import log, exp
import time
import argparse
from types import SimpleNamespace
import jack
from clock_generator import ClockGenerator

import wx
import queue

from numpy_ringbuf import Ringbuf2D, Ringbuf1D
from timetracker import TimeTracker
from data import Beat, GreedyBeat

from beatdetector import BeatDetector, SIGMA_MS
from beattracker import MIN_REL_PROMINENCE # FIXME


#gc.disable()

# usage: python main.py jack 0

p = argparse.ArgumentParser()
p.add_argument('file')
p.add_argument('start')
p.add_argument('--bpm', type=float)
p.add_argument('--offbeat', action='store_true')
p.add_argument('--duration', type=float, default=20)
p.add_argument('--step-by-step', action='store_true', default=False)
p.add_argument('--nogui', action='store_true', default=False)
p.add_argument('--plot', action='store_true', default=False)
p.add_argument('--timestep', type=float, default=1)
p.add_argument('--chunksize', type=int, default=-1)
args = p.parse_args()

if args.plot:
	import matplotlib.pyplot as plt
	import matplotlib

tt = TimeTracker()


if args.file == 'jack':
	client = jack.Client('beatdetect')
	audio_in = client.inports.register('audio_in')
	click_out = client.outports.register('click_out')
	midiclock_out = client.midi_outports.register('midiclock_out')

	total_samples = 0
	last_greedy_beat = 0.0

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
			client.outports[0].get_array()[:] = self.sine[sin_offset : sin_offset+frames] * self.click_mask[:frames] # type: ignore

	jackhandler = JackHandler(client)

	@client.set_process_callback # type: ignore[untyped-decorator]
	def process(frames: int) -> None:
		global jackhandler
		jackhandler.process(frames)

	@client.set_shutdown_callback
	def on_shutdown(status, reason): # type: ignore[no-untyped-def]
		print(f"shutting down. {reason}")
	
	@client.set_xrun_callback
	def on_xrun(usecs): # type: ignore[no-untyped-def]
		print(f"XRUN {usecs}us")

	@client.set_samplerate_callback
	def on_samplerate(samplerate): # type: ignore[no-untyped-def]
		print(f"SAMPLE RATE CHANGE {samplerate}")

	@client.set_blocksize_callback
	def on_blocksize(blocksize): # type: ignore[no-untyped-def]
		print(f"BLOCKSIZE CHANGE {blocksize}")


	samplerate = client.samplerate
	bd = BeatDetector(samplerate, force_bpm = args.bpm, timestep_desired_ms = args.timestep, time_tracker = tt)

	@dataclass
	class Tap:
		timestamp_samples: int
		samples_per_beat: int|None

	class MyFrame(wx.Frame): #type: ignore[misc]
		tap_btn: wx.Button
		jackclient: jack.Client
		taps: queue.Queue[Tap]
		tap_history: list[int] # list of timepoints [audio frames]
		labels: list[wx.StaticText]
		tempolabel: wx.StaticText
		infolabel: wx.StaticText

		def __init__(self, jackclient: jack.Client):
			wx.Frame.__init__(self, None, wx.ID_ANY, "beatdetect")

			self.tap_btn = wx.Button(self, label="tap")
			self.tap_btn.SetOwnBackgroundColour(wx.BLUE)
			self.tap_btn.Bind(wx.EVT_LEFT_DOWN,self.on_tap)
			self.jackclient = jackclient
			self.taps = queue.Queue(maxsize=100)
			self.tap_history = []

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
			if len(self.tap_history) > 0 and self.tap_history[-1] < now - 1.5*self.jackclient.samplerate:
				print(f"clearing tap history ({self.tap_history[-1]} too old for {now})")
				self.tap_history = []

			self.tap_history.append(now)

			new_samples_per_beat: int|None = None
			if len(self.tap_history) >= 2:
				new_samples_per_beat = (self.tap_history[-1] - self.tap_history[0]) // (len(self.tap_history)-1)

			self.taps.put(Tap(now, new_samples_per_beat))

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
				

	if not args.nogui:
		app = wx.App(False)
		window = MyFrame(client)
		window.Show(True)
		threading.Thread(target = lambda : app.MainLoop()).start()

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

			# our audio buffer starts at jack_frametime

			our_frametime_to_jack = jack_frametime - our_frametime
			our_frametime += frames
		
			if not args.nogui:
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
				print("sync!")

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

			if not args.nogui:
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


	exit(0)

# else

print("reading file")

data_orig, samplerate = sf.read(args.file)
t0=int(args.start)

print(len(data_orig))
print(data_orig[1])
print(samplerate)

if args.chunksize < 0: args.chunksize = samplerate

print("numpy-fying")
data_orig = np.array(data_orig)

print("trimming")
data_orig = data_orig[int(t0*samplerate): int((t0+args.duration)*samplerate), :]
print(data_orig.shape)




data = data_orig[:,0] + data_orig[:,1]
print(data.shape)


bd = BeatDetector(samplerate, plot_seconds = data.shape[0] / samplerate if args.plot else None, force_bpm = args.bpm, timestep_desired_ms = args.timestep)

for i in range(math.ceil(40 * samplerate / args.chunksize)):
	bd.process(data[i*args.chunksize:args.chunksize*(i+1)])
	if args.plot: plt.show(block=False)



#if args.plot and False: # FIXME
#	trackerax.clear()
#	trackerax.set_xlim(0, args.duration)
#	trackerax.set_ylim(-0.05, 1.05)
#	trackerax2.clear()
#	trackerax2.set_xlim(0, args.duration)
#	trackerax2.set_ylim(-0.15, 1.15)
#
#	trackerax2.scatter([b[0]*timestep_real for b in greedy_beats], [b[-1] for b in greedy_beats], color='green')
#	trackerax2.scatter([b[0]*timestep_real for b in greedy_beats], [1.07]*len(greedy_beats), color='green')
#
#	scatter_xs = []
#	scatter_ys = []
#	for t in trackers:
#		scatter_xs += [b.location*timestep_real for b in t.beats]
#		scatter_ys += [t.confidence] * len(t.beats)
#	trackerax.scatter(scatter_xs, scatter_ys, color='red')

for t in bd.trackers:
	mbt = (t.beats[-1].location - t.beats[0].location) / (len(t.beats)-1)
	mbpm = (60/mbt/bd.timestep_real)
	print("tracker suggests %.2f bpm" % mbpm)

beats=bd.trackers[0].beats # FIXME proper getter
greedy_beats = bd.greedy_beats
timestep_real = bd.timestep_real

print("%.2f%%" % (len([1 for b in beats if b.is_not_synthetic]) / len(beats)*100))


mean_beat_time = (beats[-1].location - beats[0].location) / (len(beats)-1)

if args.plot:
	axs = bd.axs
	bd.draw_plot()
	for beat in beats:
		axs[2].axhline(beat.location*timestep_real, color='red', ls='-' if beat.is_not_synthetic else '-.')
		axs[2].scatter([beat.prominence_avg,beat.prominence_avg * MIN_REL_PROMINENCE], [(beat.location+beat.time_per_beat)*timestep_real]*2, color='blue')
		spanwidth = SIGMA_MS/1000*2
		axs[2].axhspan((beat.location+beat.time_per_beat)*timestep_real-spanwidth/2, (beat.location+beat.time_per_beat)*timestep_real+spanwidth/2, color=("yellow", 0.2))
		axs[2].scatter([beat.prominence], [beat.location*timestep_real], color='red')

	for i in range(31):
		#axs[2].axhline(beats[0].location + i*mean_beat_time, color="green", ls='--')
		#axs[2].axhline((phase + i*periodicity)*timestep_real, color="purple", ls='--')
		pass

mean_bpm = (60/mean_beat_time/timestep_real)
print(f"actual mean tempo = {mean_bpm:.1f}")

errors = np.abs(([b.location for b in beats] - (beats[0].location + np.arange(len(beats)) * mean_beat_time)) * timestep_real)
errors_ms = errors*1000

print(f"beat errors: mean = {np.mean(errors_ms):.1f}ms, median = {np.median(errors_ms):.1f}ms, q90 = {np.quantile(errors_ms, 0.9):.1f}ms, max = {np.max(errors_ms):.1f}ms")
#print(f"lambda = {lam}")

if args.plot:
	fig, axs = plt.subplots(1,1, squeeze=False)
	ax: matplotlib.axes.Axes = axs[0,0]

	bpms = []
	ts = []
	for (b1, b2) in zip(beats, beats[1:]):
		t1 = b1.location * timestep_real
		t2 = b2.location * timestep_real
		bpm = 60 / (t2-t1)
		ts.append((t1+t2)/2)
		bpms.append(bpm)

	ax.plot(ts, bpms)

tt.print_stats()

print("writing out.mp3")

def write_debugout[T: Beat](filename: str, data_orig: np.ndarray, beats: list[T]) -> None:
	data_debug = data_orig.copy()
	data_debug /= np.max(data_debug)

	beep1_freq = 880
	beep2_freq = 880 * 3/2
	beep_ms = 40
	beep_fadein_ms = 0.1
	beep_fadeout_ms = 10

	beep1 = np.sin( np.arange(0, beep_ms/1000, 1/samplerate) * 2 * math.pi * beep1_freq )
	beep2 = np.sin( np.arange(0, beep_ms/1000, 1/samplerate) * 2 * math.pi * beep2_freq )
	beep2 = beep2 + beep1

	fadein = np.arange(0, 1, beep_fadein_ms/1000*samplerate)
	fadeout = np.arange(1,0, -beep_fadeout_ms/1000*samplerate)
	beep_window = np.concat([fadein, np.ones(len(beep1)-len(fadein)-len(fadeout)), fadeout])
	beep1 = beep1 * beep_window * 0.25
	beep2 = beep2 * beep_window * 0.25

	time_fixup_s = 0
	for i, beat in enumerate(beats):
		#if i%2 == 1 and t.location > 40_000: continue
		beep = beep1 if beat.is_not_synthetic else beep2
		beat.location = int((beat.location * timestep_real + time_fixup_s) * samplerate)
		if beat.location < 0: continue
		if beat.location + len(beep) >= data_debug.shape[0]: continue
		data_debug[beat.location:(beat.location+len(beep)), :] += beep.reshape(-1, 1)

	data_debug /= (1 + max(beep))

	sf.write(filename, data_debug, samplerate)

def write_beats[T: Beat](beats: list[T], filename: str) -> None:
	with open(filename, 'w') as f:
		for b in beats:
			f.write("%d\n" % b.location)

write_debugout("out.mp3", data_orig, beats)
write_debugout("out_greedy.mp3", data_orig, greedy_beats)

write_beats(beats, "out.txt")
write_beats(greedy_beats, "out_greedy.txt")

if args.plot:
	plt.show()

print("bye")
