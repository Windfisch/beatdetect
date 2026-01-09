from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Any
import math
import numpy as np
import scipy.signal as ss
import scipy.ndimage as sn
import scipy.sparse as sparse
from numpy import log, exp
from types import SimpleNamespace

from numpy_ringbuf import Ringbuf2D, Ringbuf1D
from data import Beat, GreedyBeat

from beattracker import BeatTracker
from timetracker import TimeTrackerIface, NullTimeTracker



try:
	import matplotlib.pyplot as plt
except:
	pass

def get_search_interval(trackers: Sequence[BeatTracker]) -> tuple[int, int]:
	"""Returns the smallest interval which contains all of trackers' search_intervals"""
	lo = 999999.0
	hi = -1.0
	for t in trackers:
		l, h = t.search_interval()
		lo = min(lo, l)
		hi = max(hi, h)
	return (int(lo), int(hi))

@dataclass
class Beatgrid:
	time_per_beat: float # periodicity
	beat_location: float
	peak_height: float

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
	plot: bool
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

	time_tracker: TimeTrackerIface
	"Usually the NullTimeTracker which does nothing; can be injected from outside to debug timing"

	def __init__(self, samplerate: int, timestep_desired_ms: float = 3, fft_window_ms: float = 25, cfar_avg_ms: float = 50, cfar_dead_ms: float = 10, force_bpm: None|float = None, plot_seconds: None|float = None, step_by_step: bool = False, verbose: bool = False, time_tracker: TimeTrackerIface|None = None):
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

		self.time_tracker = time_tracker if time_tracker is not None else NullTimeTracker()

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
		tt = self.time_tracker
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


