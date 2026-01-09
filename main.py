from __future__ import annotations
from dataclasses import dataclass, astuple
from typing import Self, Sequence, Any
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

MIN_REL_PROMINENCE = 0.25

if args.plot:
	import matplotlib.pyplot as plt
	import matplotlib

# FIXME MOVE beattracker.py
def get_search_interval(trackers: Sequence[BeatTracker]) -> tuple[int, int]:
	"""Returns the smallest interval which contains all of trackers' search_intervals"""
	lo = 999999.0
	hi = -1.0
	for t in trackers:
		l, h = t.search_interval()
		lo = min(lo, l)
		hi = max(hi, h)
	return (int(lo), int(hi))

tt = TimeTracker()

# FIXME move beatdetector.py
@dataclass
class Beatgrid:
	time_per_beat: float # periodicity
	beat_location: float
	peak_height: float

# FIXME MOVE data.py
# FIXME which of these are even used
@dataclass
class Beat:
	location: float
	is_not_synthetic: bool
	time_per_beat: float
	prominence_avg: float
	prominence: float

# FIXME MOVE data.py
@dataclass
class GreedyBeat(Beat):
	tracker_confidence: float
	tracker_time_per_beat: float

	@classmethod
	def from_Beat(cls, beat: Beat, tracker_confidence: float, tracker_time_per_beat: float) -> Self:
		return cls(beat.location, beat.is_not_synthetic, beat.time_per_beat, beat.prominence_avg, beat.prominence, tracker_confidence, tracker_time_per_beat)


# FIXME MOVE beattracker.py
class BeatTracker:
	# TODO
	beats: list[Beat]
	greedy_beats: list[GreedyBeat]
	timestep: float
	sigma: float
	lam: float
	time_per_beat: float
	last_prom: float
	confidence: float
	used: bool
	greedy_continuity: float # FIXME this does not really belong here

	# both beat_loc and time_per_beat are given in timesteps, i.e. not neccessarily msec
	def __init__(self, timestep: float, sigma: float, lam: float, beat_loc: float, time_per_beat: float, confidence: float, last_prom: float, beats: list[Beat]):
		self.timestep = timestep
		self.sigma = sigma
		self.lam = lam
		self.beat_loc = beat_loc
		self.time_per_beat = time_per_beat
		self.last_prom = last_prom
		self.confidence = confidence
		self.beats = beats[-10:] # limit beat array length
		self.used = False

	# FIXME is this even used outside of greedy tracking?
	def tpb(self) -> float:
		if len(self.beats) > 2:
			first = max(0, len(self.beats) - 1 - 8)
			last = len(self.beats)-1
			return (self.beats[last].location - self.beats[first].location) / (last-first)
		else:
			return self.time_per_beat

	# returns interval where beats are expected as (begin, end) tuple; unit = timesteps
	def search_interval(self) -> tuple[float, float]:
		expected_loc = self.beat_loc + self.time_per_beat
		win_timesteps = self.sigma * 4
		return (expected_loc - win_timesteps, expected_loc+win_timesteps)

	def next_beat(self, peak_locs: Sequence[float], peak_prominences: Sequence[float]) -> list[BeatTracker]:
		if self.time_per_beat < 50/1000/self.timestep:
			# kill degenerate beat detectors that suggest an infinite tempo
			return []

		expected_loc = self.beat_loc + self.time_per_beat

		@dataclass
		class Peak:
			relevance: float  # normalized posterior probability
			location: float   # beat location in timesteps
			is_not_synthetic: bool
			prominence: float # peak prominence

		peaks: list[Peak] = []
		for loc, prom in zip(peak_locs, peak_prominences):
			t_diff = loc - expected_loc
			relevance = math.exp(-0.5*(t_diff/self.sigma)**2) / (self.lam * math.exp(-self.lam * prom))
			if loc > self.beat_loc + self.time_per_beat*0.1:
				peaks.append(Peak(
					relevance=relevance,
					location=loc,
					is_not_synthetic=True,
					prominence=prom))

		peaks.append(Peak(
			relevance=1 / (self.lam * math.exp(-self.lam * self.last_prom * MIN_REL_PROMINENCE)),
			location=expected_loc,
			is_not_synthetic=False,
			prominence=0))

		# normalize peak relevances (to sum=1)
		relevance_sum = sum([p.relevance for p in peaks])
		for p in peaks: p.relevance /= relevance_sum

		peaks = sorted(peaks, key=lambda p: p.relevance)[::-1]

		alpha = 0
		confidence = self.confidence * (1-alpha) + 1 * alpha

		tpb_alpha = 0.7 # 0.98
		prom_alpha = 0.5
		result: list[BeatTracker] = []
		for p in peaks:
			tpb_new = tpb_alpha * self.time_per_beat + (1-tpb_alpha)*(p.location - self.beats[-1].location)
			conf_new = confidence * p.relevance
			prom_new = prom_alpha*self.last_prom + (1-prom_alpha)*p.prominence
			result.append(BeatTracker(self.timestep, self.sigma, self.lam, p.location, tpb_new, conf_new, prom_new, self.beats + [Beat(p.location, p.is_not_synthetic, tpb_new, prom_new, p.prominence)]))
		return result

SIGMA_MS=30 # 30 is good for most stuff

class BeatDetector:
	samplerate: int
	force_bpm: float | None
	verbose: bool
	samplestep: int
	timestep_real: float
	fft_window_size: int
	fft_window: np.ndarray
	fft_size: int
	logbins: float
	log_binning: LogBinning
	cfar_kernel: np.ndarray
	plot: float | None
	step_by_step: bool
	audio_history: np.ndarray
	fft_history: np.ndarray
	snr_history: Ringbuf2D
	snrsum_history: Ringbuf1D
	smoothing_sigma_ms: float
	next_sample: int
	next_peakstat_sample: int
	next_timestep: int
	need_tempo: bool
	trackers: list[BeatTracker]
	beats: list[Beat]
	greedy_beats: list[GreedyBeat]
	plots: SimpleNamespace
	axs: Any
	peakstataxs: Any

	def __init__(self, samplerate: int, timestep_desired_ms: float = 3, fft_window_ms: float = 25, cfar_avg_ms: float = 50, cfar_dead_ms: float = 10, force_bpm: None|float = None, plot_seconds: None|float = None, step_by_step: bool = False, verbose: bool = False):
		self.samplerate = samplerate
		self.force_bpm = force_bpm
		self.verbose = verbose
	
		timestep_desired = timestep_desired_ms / 1000
		self.samplestep = int(samplerate * timestep_desired)
		self.timestep_real = self.samplestep / samplerate
		print(f"using a timestep of {self.timestep_real*1000:.3f}ms")
		
		self.fft_window_size = int(fft_window_ms / 1000 * samplerate)
		self.fft_window = np.hamming(self.fft_window_size)
		self.fft_size = 2 * self.fft_window_size
		fft_output_size = int(self.fft_size/2) + 1
		self.logbins = math.log(self.fft_size, 2)*12 # TODO does not need to be a member # FIXME FIXME FIXME this should use fft_output_size, not fft_size!
		self.log_binning = LogBinning(self.logbins, fft_output_size)
		
		self.cfar_kernel = cfar_kernel( int(cfar_avg_ms / 1000 / self.timestep_real), int(cfar_dead_ms / 1000 / self.timestep_real) )
		print("kernel len = ", len(self.cfar_kernel))

		self.plot = plot_seconds is not None
		self.step_by_step = step_by_step

		self.audio_history = np.zeros(0)
		self.fft_history = np.zeros((len(self.cfar_kernel)-1, int(self.logbins)))
		self.snr_history = Ringbuf2D(samplerate*30, int(self.logbins))
		self.snrsum_history = Ringbuf1D(samplerate*30)

		self.smoothing_sigma_ms = 10

		self.next_sample = 0
		self.next_peakstat_sample = 0
		self.next_timestep = 0
		self.need_tempo = True

		self.trackers = []

		self.beats = []
		self.greedy_beats = []

		if self.plot:
			assert plot_seconds is not None
			plot_timesteps = int(plot_seconds / self.timestep_real)
			self.plots = SimpleNamespace()
			self.plots.waterfall = np.ndarray((plot_timesteps, int(self.logbins)))
			self.plots.snr = np.ndarray((plot_timesteps,int(self.logbins)))
			self.plots.snrsum = np.ndarray(plot_timesteps)
			self.plots.lambdas = []
			self.plots.tracker_respawns = []

			fig, axs1 = plt.subplots(1,3)
			fig, axs2 = plt.subplots(1,3)
			self.axs = np.ndarray.flatten(np.concat([axs1,axs2]))
			fig2 = plt.figure()
			self.peakstatax = fig2.add_subplot()

	def sample_from_timestep(self, ts: float) -> float:
		#self.samplestep == self.timestep_real * self.samplerate
		#self.fft_window_size = int(fft_window_ms / 1000 * samplerate)
		return ts * self.samplestep + self.fft_window_size / 2

	def timestep_from_sample(self, s: float) -> float:
		return (s - self.fft_window_size/2) / self.samplestep
	
	def deltatimestep_from_deltasample(self, s: float) -> float:
		return s / self.samplestep
	
	def seconds_from_timestep(self, ts: float) -> float:
		return self.sample_from_timestep(ts) / self.samplerate

	def draw_plot(self) -> None:
		fig, ax = plt.subplots(1,1)
		ax.plot([x for x,y in self.plots.lambdas], [y for x,y in self.plots.lambdas])
		for a,b in self.plots.tracker_respawns:
			ax.axvspan(a,b, color=("yellow", 0.3))

		axs = self.axs
		axs[0].clear()
		axs[1].clear()
		axs[2].clear()
		axs[0].imshow(self.plots.waterfall, aspect='auto', extent=[0,self.plots.waterfall.shape[1],self.plots.waterfall.shape[0]*self.timestep_real,0]) #, vmax=1e+4, vmin=0)
		axs[1].imshow(self.plots.snr, aspect='auto', vmin=0, vmax=np.quantile(self.plots.snr, 0.97)*0.8, extent=[0,self.plots.snr.shape[1],self.plots.snr.shape[0]*self.timestep_real,0])
		axs[1].sharex(axs[0])
		axs[1].sharey(axs[0])
		axs[2].plot(self.snrsum_history.get(), np.arange(len(self.snrsum_history.get()))*self.timestep_real, color='orange')
		axs[2].plot(sn.gaussian_filter1d(self.snrsum_history.get(), self.smoothing_sigma_ms/1000/self.timestep_real), np.arange(len(self.snr_history.get()))*self.timestep_real, color='blue')
		axs[2].sharey(axs[0])

	def resync(self, timestep: float, new_tpb: float|None = None) -> None:
		if len(self.trackers) > 0:
			tracker = self.trackers[0]
			beat = tracker.beats[-1]

			print(f"shifting next beat from {beat.location} to {timestep} while {'retaining' if new_tpb is None else 'updating'} tpb={new_tpb}; formerly {tracker.time_per_beat}={beat.time_per_beat}")
			newbeat = Beat(
				location = timestep,
				is_not_synthetic = False,
				time_per_beat = new_tpb if new_tpb is not None else beat.time_per_beat,
				prominence_avg = beat.prominence_avg,
				prominence = beat.prominence
			)
			tracker.beats = [newbeat]
			if new_tpb is not None:
				tracker.time_per_beat = new_tpb
			tracker.beat_loc = timestep
			self.trackers = [tracker]
	
	def update_greedy(self) -> None:
			PUNISH_DISCONTINOUS = 0.5

			if len(self.greedy_beats) > 0:
				last, tpb = self.greedy_beats[-1].location, self.greedy_beats[-1].tracker_time_per_beat
				for tr in self.trackers:
					rel_distance = ((tr.beats[-1].location - last) / tpb + 0.5) % 1 - 0.5
					tr.greedy_continuity = PUNISH_DISCONTINOUS * np.exp(- (rel_distance / 0.1) ** 2) + 1-PUNISH_DISCONTINOUS
			else:
				for tr in self.trackers:
					tr.greedy_continuity = 1

			best = max(self.trackers, key = lambda tr : tr.confidence * tr.greedy_continuity)
			if best.used == False:
				self.greedy_beats.append(GreedyBeat.from_Beat(best.beats[-1], best.confidence, best.tpb())) # FIXME .tpb is wrong here?
				self.greedy_beats = self.greedy_beats[-10:]
			
			for tr in self.trackers:
				tr.used = True

	def process(self, audio: np.ndarray) -> None:
		tt.begin('concat audio history')
		first_sample = self.next_sample
		self.next_sample += len(audio)
		if self.verbose: print(f"consuming {len(audio)} samples, starting at {first_sample}")

		data = np.concatenate([self.audio_history, audio])
		# audio[-1] is at t = self.next_sample samples
		# so is data[-1]

		tt.begin("overlapping windows")
		x = overlapping_windows(data, self.fft_window, self.samplestep)
		self.next_timestep += x.shape[0]
		tt.begin('slice audio history')
		self.audio_history = data[self.samplestep * x.shape[0] : ]
		if self.verbose: print(f"got {x.shape[0]} new timesteps; audio history len = {len(self.audio_history)}")

		tt.begin('fft')
		y = np.absolute( np.fft.rfft(x, n = self.fft_size, axis=1) )
		y[:, 0:10] = np.sum( y[:, 0:10], axis=1 ).reshape(-1,1) / 10 # sum bass together

		tt.begin('binning')
		if self.verbose: print("binning")
		y = self.log_binning.log_sum2(y)

		y = np.log10(y + 1e-4) * 20

		waterfall = y

		# x[-1] and y[-1] is at t = self.next_timestep timesteps

		tt.begin('concat fft history')
		self.fft_history = np.concatenate([self.fft_history[-(len(self.cfar_kernel)-1):, :], y], axis=0)
		if self.verbose: print(f"fft history len = {self.fft_history.shape[0]} timesteps")
	
		tt.begin('cfar') # FIXME this scales not so well
		if self.verbose: print("convolving")
		if self.fft_history.shape[0] >= self.cfar_kernel.shape[0]:
			# both options do exacty the same, but their performance scales differently
			# with number_of_iterations and batchsize.
			# a takeover between 15 and 20 looks legit on my machine and limits worst-case
			# time to approx. 350%, which is still only 60% of fft runtime.
			if self.fft_history.shape[0]-self.cfar_kernel.shape[0] >= 15:
				y = np.array([
					np.convolve(col, self.cfar_kernel, mode='valid')
					for col in self.fft_history.T
				]).T
			else:
				y = ss.convolve2d(self.fft_history, self.cfar_kernel[:, np.newaxis], 'valid')
		else:
			if self.verbose: print("not enough fft history to perform a cfar, returning")
			return

		assert y.shape[0] == waterfall.shape[0]

		tt.begin('aftermath')
		if self.verbose: print("maximum")
		y = np.fmax(y - 0, 0)

		if self.plot:
			self.plots.waterfall[self.next_timestep-waterfall.shape[0] : self.next_timestep, :] = waterfall


		if self.verbose: print("done")

		z = np.sum( y, axis=1)

		tt.begin('snr(sum) history')
		self.snr_history.append(y)
		self.snrsum_history.append(z)
		
		# x[-1], y[-1], z[-1], self.snr_history.get()[-1] and self.snrsum_history.get()[-1] is at t = self.next_timestep timesteps

		if self.verbose: print(f"snr history len = {self.snr_history.get().shape[0]} timesteps")

		if self.plot:
			self.plots.snr[self.next_timestep-self.snr_history.get().shape[0] : self.next_timestep, :] = self.snr_history.get()
			self.plots.snrsum[self.next_timestep-self.snrsum_history.get().shape[0] : self.next_timestep] = self.snrsum_history.get()
		
		if self.next_sample >= self.next_peakstat_sample:
			# peak stats and tempo estimation are only done every second or so (more rarely if larger chunks of audio are fed into the algorithm!)
			self.next_peakstat_sample = self.next_sample + int(self.samplerate / 2)

			tt.begin('peak stats') # FIXME scales not so well
			if self.verbose: print("Running statistics on the peaks")
			peaks_result = ss.find_peaks(sn.gaussian_filter1d(self.snrsum_history.get(), self.smoothing_sigma_ms/1000/self.timestep_real), height=0, distance = (0.01 / self.timestep_real), prominence=0) # FIXME distance?? remove
			prominences = sorted(peaks_result[1]['prominences'])[::-1]
			if len(prominences) > 0:
				lam = float(1/np.mean(prominences))
				if self.plot:
					self.plots.lambdas.append((self.next_timestep-1, lam))
				if self.verbose: print("lambda = %.6f" % lam)

				if self.need_tempo:
					tt.begin('tempo estimation')

					crop_amount = 0

					history_first = self.next_timestep - self.snr_history.get().shape[0]
					HISTORY_DROP = self.cfar_kernel.shape[0] + 1

					drop = max(0, HISTORY_DROP - history_first)

					if drop > 0:
						print(f"dropping {drop} frames from history because of warmup")

					tempo_result, tempo_missing = self.estimate_tempo_and_phase(self.snr_history.get()[drop:,:], self.force_bpm, self.next_timestep - len(self.snrsum_history.get()))
					if tempo_missing > 0:
						print(f"Tempo estimation pending, need {tempo_missing} more samples")
					else:
						assert tempo_result is not None
						location = tempo_result.beat_location + (self.next_timestep - len(self.snrsum_history.get()))
						tempo = 60/tempo_result.time_per_beat/self.timestep_real
						print(f"Tempo estimated -> {tempo} bpm with beat at timestep {location}")

						if self.plot:
							self.plots.tracker_respawns.append((self.next_timestep-len(self.snrsum_history.get()), self.next_timestep))
						
						self.trackers = [BeatTracker(
							self.timestep_real,
							SIGMA_MS/1000/self.timestep_real,
							lam,
							location,
							tempo_result.time_per_beat,
							1,
							tempo_result.peak_height,
							[Beat(
								location=location,
								is_not_synthetic=False,
								time_per_beat=tempo_result.time_per_beat,
								prominence_avg=tempo_result.peak_height,
								prominence=tempo_result.peak_height
							)]
						)]

						self.need_tempo = False

				if self.plot:
					self.peakstatax.plot(prominences, np.arange(len(prominences))/len(prominences))
					self.peakstatax.plot(np.arange(max(prominences)), np.exp(-lam * np.arange(max(prominences)) ))
			else:
				if self.verbose: print("ehhh too few prominences")

		tt.begin('beat tracking')

		window_start, window_end = 0, 0

		smoothing_sigma_timesteps = self.smoothing_sigma_ms / 1000 / self.timestep_real
		smoothing_context_timesteps = int(math.ceil(3 * smoothing_sigma_timesteps))
		while len(self.trackers) > 0:
			if all([tr.search_interval()[1] > window_end for tr in self.trackers]):
				if self.verbose: print("============================")
				window_start, window_end = get_search_interval(self.trackers)
				window_start = max(window_start, self.next_timestep - len(self.snrsum_history.get()) + smoothing_context_timesteps)
				window_end = max(window_end, 0)
				if window_end + smoothing_context_timesteps >= self.next_timestep:
					if self.verbose: print("not enough audio, exiting")
					break

				if self.verbose: print(f"window = {window_start}..{window_end}, len = {window_end-window_start}")
				window = self.snrsum_history.get()[
					window_start-(self.next_timestep-len(self.snrsum_history.get())) - smoothing_context_timesteps :
					window_end  -(self.next_timestep-len(self.snrsum_history.get())) + smoothing_context_timesteps
				]
				window = sn.gaussian_filter1d(window, smoothing_sigma_timesteps)
				window = window[smoothing_context_timesteps : -smoothing_context_timesteps]
				assert len(window) == window_end - window_start
				if len(window) == 0:
					if self.verbose: print("window is empty")
					break

				result = ss.find_peaks(window, height=0, distance = (0.01 / self.timestep_real), prominence=0.05 * np.max(window))

				peaks = result[0] + window_start
				heights = result[1]['prominences']
				if self.verbose: print(f"found {len(peaks)} peaks")
				if len(peaks) == 0: continue

			if self.verbose: print("------------------------------------")
			trackers_new: list[BeatTracker] = []
			n_updated = 0
			for tr in self.trackers:
				lo, hi = tr.search_interval()
				if int(hi) <= window_end:
					trackers_new += tr.next_beat(peaks, heights)
					n_updated += 1
				else:
					trackers_new.append(tr)

			if n_updated == 0:
				if self.verbose: print("no tracker was updated, exiting")
				if self.verbose: print(f"(window = {window_start}..{window_end}, len = {window_end-window_start})")
				if self.verbose: print([tr.search_interval() for tr in trackers_new])
				assert False # FIXME

			DEDUP_LOC_THRESHOLD_MS = 10 # FIXME or 50?
			trackers_new.sort(key = lambda t : -t.confidence)
			trackers_dedup = []
			trackers_new2: list[BeatTracker|None] = list(trackers_new) # FIXME unneccessary clone
			for i in range(len(trackers_new2)):
				tr_: BeatTracker|None = trackers_new2[i]
				if tr_ is None: continue
				tr = tr_
				loc = tr.beats[-1].location * self.timestep_real * 1000

				for j in range(i+1, len(trackers_new2)):
					candidate = trackers_new2[j]
					if candidate is None: continue
					candidate_loc = candidate.beats[-1].location * self.timestep_real * 1000

					if abs(loc - candidate_loc) <= DEDUP_LOC_THRESHOLD_MS:
						trackers_new2[j] = None
						tr.confidence += candidate.confidence

				trackers_dedup.append(tr)

			if self.verbose: print(f"deduplication removed {len(trackers_new2)-len(trackers_dedup)} of {len(trackers_new2)} trackers")
			self.trackers = trackers_dedup

			self.trackers.sort(key = lambda t : -t.confidence)

			self.trackers = self.trackers[0:10]

			sum_conf = sum([t.confidence for t in self.trackers])
			for t in self.trackers: t.confidence /= sum_conf


			if self.verbose: print("confidences: ", ", ".join(["%5f" % t.confidence for t in self.trackers]))

			self.update_greedy()

#			if self.plot and args.step_by_step:
#				trackerax.clear()
#				trackerax.set_xlim(0, args.duration)
#				trackerax.set_ylim(-0.05, 1.05)
#				trackerax2.clear()
#				trackerax2.set_xlim(0, args.duration)
#				trackerax2.set_ylim(-0.15, 1.15)
#
#				trackerax2.scatter([b.location*self.timestep_real for b in greedy_beats], [b[-1] for b in greedy_beats], color='green')
#				trackerax2.scatter([b.location*self.timestep_real for b in greedy_beats], [1.07]*len(greedy_beats), color='green')
#
#				scatter_xs = []
#				scatter_ys = []
#				for t in self.trackers:
#					scatter_xs += [b.location*self.timestep_real for b in t.beats]
#					scatter_ys += [t.confidence] * len(t.beats)
#
#				trackerax.scatter(scatter_xs_old, scatter_ys_old, color='gray')
#				trackerax.scatter(scatter_xs, scatter_ys, color='red')
#				scatter_xs_old = scatter_xs
#				scatter_ys_old = scatter_ys
#				if args.step_by_step:
#					plt.ginput()

		tt.begin('done')


	# Returns either (a Beatgrid, 0) or (None, rows needed to compute a beatgrid)
	def estimate_tempo_and_phase(self, snr_history: np.ndarray, force_bpm:float|None = None, plot_time_offset: float = 0) -> tuple[Beatgrid|None, int]:
		min_bpm = 60
		max_bpm = 300
		correlation_window = int(5/self.timestep_real)
		rows_needed = correlation_window + int((60/min_bpm)/self.timestep_real)
		if snr_history.shape[0] < rows_needed:
			return None, rows_needed - snr_history.shape[0]

		print("Got enough data to compute a tempo estimate!")

		y = snr_history # FIXME naming
		print(y.shape)
		equalizer = 1 - 0*(((np.arange(y.shape[1]) / y.shape[1]) - 0.5) * 2)
		equalizer = equalizer[None, :]
		print(equalizer)
		z = sn.gaussian_filter1d(np.sum( y * equalizer , axis=1), self.smoothing_sigma_ms/1000/self.timestep_real) # FIXME naming

		#corr_data = y[ -int((60/min_bpm) / timestep_real)-correlation_window:, :]
		#corr_reference = corr_data[-correlation_window:, :]
		#correlation = np.apply_along_axis( lambda foo: np.flip(ss.correlate(
		#	foo[ -int((60/min_bpm) / timestep_real)-correlation_window:],
		#	foo[-correlation_window:], 'valid')), axis=0, arr=y)
		correlation = np.apply_along_axis( lambda foo: np.flip(ss.correlate(
			foo[ -(int((60/min_bpm) / self.timestep_real)+correlation_window) : ],
			foo[ -correlation_window : ],
			'valid')), axis=0, arr=y)

		correlation_1d = np.sum(correlation, axis=1)

		crop = int(60/max_bpm/self.timestep_real)

		peak_limit = (np.quantile(correlation_1d[crop:], 0.9))

		peaks = ss.find_peaks(correlation_1d, height=peak_limit, prominence=peak_limit*0.3)[0]
		tempi = [ (60/(t*self.timestep_real)) for t in peaks]
		print("possible tempi: " + ", ".join(["%.1fbpm" % t for t in tempi]))

		if force_bpm is None:
			tempo = tempi[0]
			if tempo > 240: tempo /= 2
		else:
			tempo = force_bpm
			print("Using bpm override")

		print(f"Using tempo {tempo}")


		periodicity = int(np.round(60/tempo/self.timestep_real))

		if self.plot:
			axs = self.axs
			axs[3].imshow(correlation, aspect='auto', extent=[0,correlation.shape[1],correlation.shape[0]*self.timestep_real,0])
			axs[4].set_xlim(xmin=0, xmax=np.max(correlation_1d[crop:]))
			axs[4].plot(correlation_1d, np.arange(len(correlation_1d))*self.timestep_real)
			axs[4].axvline(peak_limit)

			axs[4].axhline(periodicity*self.timestep_real)
			axs[3].axhline(periodicity*self.timestep_real)

			axs[4].sharey(axs[3])
			
			for tempo_ in tempi:
				periodicity_ = int(np.round(60/tempo_/self.timestep_real))
				phase_window = int((5 / self.timestep_real) / periodicity_)*periodicity_
				n = phase_window / periodicity_
				phases = z[:phase_window].reshape(-1, periodicity_).sum(axis=0) / n
				axs[5].plot((np.arange(len(phases))+plot_time_offset)*self.timestep_real, phases, lw=1)

		phase_window = int((5 / self.timestep_real) / periodicity)*periodicity
		n = phase_window / periodicity
		phases = z[:phase_window].reshape(-1, periodicity).sum(axis=0) / n
		if self.plot: axs[5].plot(np.arange(len(phases))*self.timestep_real, phases)

		phase: int = int(np.argmax(phases))
		return Beatgrid(time_per_beat=periodicity, beat_location=phase, peak_height=phases[phase]), 0

def cfar_kernel(n: int, guard: int) -> np.ndarray:
	return np.array( [1] + [0]*guard + [-1/n]*n )

def log_avg(a: np.ndarray, bins: int) -> np.ndarray:
	l = log(a.shape[1]) / bins
	bins = int(bins)
	result=np.zeros((a.shape[0], bins))

	for i in range(bins):
		begin = int(exp(i*l))
		end = int(exp((i+1)*l))
		if end <= begin: end=begin+1

		result[:, i] = np.sum(a[:, begin:end], axis=1) / (end-begin)
	
	return result

def log_avg2(a: np.ndarray, bins: float) -> np.ndarray:
	l = log(a.shape[1]) / bins
	bins = int(bins)
	result=np.zeros((a.shape[0], bins))

	for i in range(bins):
		begin = int(exp(i*l))
		end = int(exp((i+1)*l))
		if end <= begin: end=begin+1

		result[:, i] = np.log( np.sum( np.exp(a[:, begin:end]), axis=1) / (end-begin) )
	
	return result

class LogBinning:
	def __init__(self, bins: float, fftsize: int):
		self.borders = np.logspace(0, np.log10(fftsize)*int(bins)/bins, int(bins)+1).astype(int)
		self.borders_end = np.maximum(self.borders[1:], self.borders[:-1]+1)
		matrix = np.zeros((fftsize, int(bins)))

		for i in range(int(bins)):
			begin = self.borders[i]
			end = self.borders_end[i]
			for j in range(begin, end):
				matrix[j, i] = 1/(end-begin)
		self.matrix = sparse.csr_array(matrix) # doesn't really matter if csr or csc

	def log_sum2(self, a: np.ndarray) -> np.ndarray:
		return a @ self.matrix # type: ignore[no-any-return]

def overlapping_windows(data: np.ndarray, window: np.ndarray, step: int) -> np.ndarray:
	if len(data) < len(window):
		return np.zeros((0, len(window)))
	N = len(window)
	return np.lib.stride_tricks.as_strided(data,( (len(data)-len(window)) // step + 1, N), [data.strides[0]*step, data.strides[0]]) * window # type: ignore[no-any-return]

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
	bd = BeatDetector(samplerate, force_bpm = args.bpm, timestep_desired_ms = args.timestep)

	@dataclass
	class Tap:
		timestamp_samples: int
		samples_per_beat: int|None

	class MyFrame(wx.Frame):
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
