import soundfile as sf
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

from numpy_ringbuf import Ringbuf2D, Ringbuf1D
from timetracker import TimeTracker

p = argparse.ArgumentParser()
p.add_argument('file')
p.add_argument('start')
p.add_argument('--bpm', type=float)
p.add_argument('--offbeat', action='store_true')
p.add_argument('--duration', type=float, default=20)
p.add_argument('--step-by-step', action='store_true', default=False)
p.add_argument('--plot', action='store_true', default=False)
p.add_argument('--timestep', type=float, default=1)
p.add_argument('--chunksize', type=int, default=-1)
args = p.parse_args()

MIN_REL_PROMINENCE = 0.25

if args.plot:
	import matplotlib.pyplot as plt

def get_search_interval(trackers):
	lo = 999999
	hi = -1
	for t in trackers:
		l, h = t.search_interval()
		lo = min(lo, l)
		hi = max(hi, h)
	return (int(lo), int(hi))

serial_number = 1

tt = TimeTracker()

class BeatTracker:
	# both beat_loc and time_per_beat are given in timesteps, i.e. not neccessarily msec
	def __init__(self, parent, timestep, sigma, lam, beat_loc, time_per_beat, confidence, last_prom, beats):
		global serial_number
		self.timestep = timestep
		self.sigma = sigma
		self.parent = parent
		self.serial = serial_number
		serial_number+=1
		self.lam = lam
		self.beat_loc = beat_loc
		self.time_per_beat = time_per_beat
		self.last_prom = last_prom
		self.confidence = confidence
		self.beats = beats[:]
		self.used = False
		if confidence > 0.05:
			#print(f"Tracker #{self.parent} spawns #{self.serial} at {self.beat_loc} with {confidence*100:.2f}%, expected = {self.time_per_beat+self.beat_loc:.2f}, last_prom = {self.last_prom}, tpb = {self.time_per_beat:.2f}")
			pass

	def search_interval(self):
		expected_loc = self.beat_loc + self.time_per_beat
		win_timesteps = self.sigma * 4
		return (expected_loc - win_timesteps, expected_loc+win_timesteps)

	def next_beat(self, peak_locs, peak_prominences):
		if self.time_per_beat < 50/1000/self.timestep:
			# kill degenerate beat detectors that suggest an infinite tempo
			return []


		expected_loc = self.beat_loc + self.time_per_beat
		peaks = []

		for loc, prom in zip(peak_locs, peak_prominences):
			t_diff = loc - expected_loc
			relevance = math.exp(-0.5*(t_diff/self.sigma)**2) / (self.lam * math.exp(-self.lam * prom))
			if loc > self.beat_loc + self.time_per_beat*0.1:
				peaks.append((relevance, loc, True, prom))

		#print(f"expected = {expected_loc}, max = {max(peaks+[(-999,0, False, 0)])}, lastprom = {self.last_prom}, relevance = {1 / (self.lam * math.exp(-self.lam * self.last_prom))}")
		peaks.append((1 / (self.lam * math.exp(-self.lam * self.last_prom * MIN_REL_PROMINENCE)), expected_loc, False, 0))

		relevance_sum = sum([r for r,l,f,p in peaks])
		peaks = [(r/relevance_sum, l, f,p) for r,l,f,p in peaks]
		peaks = sorted(peaks)[::-1]

		#print("sorted: ", peaks)

		alpha = 0
		confidence = self.confidence * (1-alpha) + 1 * alpha

		tpb_alpha = 0.7
		prom_alpha = 0.5
		result = []
		for rel, loc, found, prom in peaks:
			tpb_new = tpb_alpha * self.time_per_beat + (1-tpb_alpha)*(loc - self.beats[-1][0])
			conf_new = confidence * rel
			#print(f"spawning found={found}, rel={rel}, conf={conf_new}, prom={prom}")
			prom_new = prom_alpha*self.last_prom + (1-prom_alpha)*prom
			result.append(BeatTracker(self.serial, self.timestep, self.sigma, self.lam, loc, tpb_new, conf_new, prom_new, self.beats + [(loc, found, tpb_new, prom_new, prom)]))
		return result

SIGMA_MS=30 # 30 is good for most stuff

class BeatDetector:
	def __init__(self, samplerate, timestep_desired_ms = 3, fft_window_ms=25, cfar_avg_ms=50, cfar_dead_ms=10, force_bpm=None, plot_seconds = None, step_by_step=False, verbose=False):
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

	def sample_from_timestep(self, ts):
		#self.samplestep == self.timestep_real * self.samplerate
		#self.fft_window_size = int(fft_window_ms / 1000 * samplerate)
		return ts * self.samplestep + self.fft_window_size / 2

	def timestep_from_sample(self, s):
		return (s - self.fft_window_size/2) / self.samplestep
	
	def seconds_from_timestep(self, ts):
		return self.sample_from_timestep(ts) / self.samplerate

	def draw_plot(self):
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

	def process(self, audio):
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

		z = np.sum( y[:, 0:], axis=1)

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
			self.next_peakstat_sample = self.next_sample + self.samplerate / 2

			tt.begin('peak stats') # FIXME scales not so well
			if self.verbose: print("Running statistics on the peaks")
			result = ss.find_peaks(sn.gaussian_filter1d(self.snrsum_history.get(), self.smoothing_sigma_ms/1000/self.timestep_real), height=0, distance = (0.01 / self.timestep_real), prominence=0) # FIXME distance?? remove
			prominences = sorted(result[1]['prominences'])[::-1]
			if len(prominences) > 0:
				lam = 1/np.mean(prominences)
				if self.plot:
					self.plots.lambdas.append((self.next_timestep-1, lam))
				if self.verbose: print("lambda = %.6f" % lam)

				if self.need_tempo:
					tt.begin('tempo estimation')
					result, missing = self.estimate_tempo_and_phase(self.snr_history.get(), self.snrsum_history.get(), self.force_bpm)
					if missing > 0:
						print(f"Tempo estimation pending, need {missing} more samples")
					else:
						periodicity, phase, amplitude = result
						phase += (self.next_timestep - len(self.snrsum_history.get()))
						tempo = 60/periodicity/self.timestep_real
						print(f"Tempo estimated -> {tempo} bpm with beat at timestep {phase}")

						if self.plot:
							self.plots.tracker_respawns.append((self.next_timestep-len(self.snrsum_history.get()), self.next_timestep))
						self.trackers = [BeatTracker(0, self.timestep_real, SIGMA_MS/1000/self.timestep_real, lam, phase, periodicity, 1, amplitude, [(phase, False, periodicity, amplitude, amplitude)])]

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
			trackers_new = []
			n_updated = 0
			for tr in self.trackers:
				lo, hi = tr.search_interval()
				if int(hi) <= window_end:
					trackers_new += tr.next_beat(peaks, heights)
					n_updated += 1
				else:
					trackers_new.append(tr)
			self.trackers = trackers_new

			if n_updated == 0:
				if self.verbose: print("no tracker was updated, exiting")
				if self.verbose: print(f"(window = {window_start}..{window_end}, len = {window_end-window_start})")
				if self.verbose: print([tr.search_interval() for tr in self.trackers])
				assert False # FIXME

			DEDUP_LOC_THRESHOLD_MS = 10
			self.trackers.sort(key = lambda t : -t.confidence)
			trackers_dedup = []
			for i in range(len(self.trackers)):
				tr = self.trackers[i]
				if tr is None: continue
				loc = tr.beats[-1][0] * self.timestep_real * 1000

				for j in range(i+1, len(self.trackers)):
					candidate = self.trackers[j]
					if candidate is None: continue
					candidate_loc = candidate.beats[-1][0] * self.timestep_real * 1000

					if abs(loc - candidate_loc) <= DEDUP_LOC_THRESHOLD_MS:
						self.trackers[j] = None
						tr.confidence += candidate.confidence

				trackers_dedup.append(tr)

			if self.verbose: print(f"deduplication removed {len(self.trackers)-len(trackers_dedup)} of {len(self.trackers)} trackers")
			self.trackers = trackers_dedup

			self.trackers.sort(key = lambda t : -t.confidence)

			self.trackers = self.trackers[0:10]
			
			sum_conf = sum([t.confidence for t in self.trackers])
			for t in self.trackers: t.confidence /= sum_conf


			if self.verbose: print("confidences: ", ", ".join(["%5f" % t.confidence for t in self.trackers]))

			if self.trackers[0].used == False:
				self.greedy_beats.append(self.trackers[0].beats[-1] + (self.trackers[0].confidence,))
			for tr in self.trackers:
				tr.used = True

			if self.plot and args.step_by_step:
				trackerax.clear()
				trackerax.set_xlim(0, args.duration)
				trackerax.set_ylim(-0.05, 1.05)
				trackerax2.clear()
				trackerax2.set_xlim(0, args.duration)
				trackerax2.set_ylim(-0.15, 1.15)

				trackerax2.scatter([b[0]*self.timestep_real for b in greedy_beats], [b[-1] for b in greedy_beats], color='green')
				trackerax2.scatter([b[0]*self.timestep_real for b in greedy_beats], [1.07]*len(greedy_beats), color='green')

				scatter_xs = []
				scatter_ys = []
				for t in self.trackers:
					scatter_xs += [b[0]*self.timestep_real for b in t.beats]
					scatter_ys += [t.confidence] * len(t.beats)

				trackerax.scatter(scatter_xs_old, scatter_ys_old, color='gray')
				trackerax.scatter(scatter_xs, scatter_ys, color='red')
				scatter_xs_old = scatter_xs
				scatter_ys_old = scatter_ys
				if args.step_by_step:
					plt.ginput()

		tt.begin('done')


	def estimate_tempo_and_phase(self, snr_history, snrsum_history, force_bpm=None):
		min_bpm = 60
		max_bpm = 300
		y = snr_history # FIXME naming
		z = snrsum_history # FIXME naming

		correlation_window = int(5/self.timestep_real)
		rows_needed = correlation_window + int((60/min_bpm)/self.timestep_real)
		if snr_history.shape[0] < rows_needed:
			return None, rows_needed - snr_history.shape[0]

		print("Got enough data to compute a tempo estimate!")
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
				axs[5].plot(np.arange(len(phases))*self.timestep_real, phases, lw=1)

		phase_window = int((5 / self.timestep_real) / periodicity)*periodicity
		n = phase_window / periodicity
		phases = z[:phase_window].reshape(-1, periodicity).sum(axis=0) / n
		if self.plot: axs[5].plot(np.arange(len(phases))*self.timestep_real, phases)

		phase = np.argmax(phases)
		return (periodicity, phase, phases[phase]), 0

def cfar_kernel(n, guard):
	return np.array( [1] + [0]*guard + [-1/n]*n )

def log_avg(a, bins):
	l = log(a.shape[1]) / bins
	bins = int(bins)
	result=np.zeros((a.shape[0], bins))

	for i in range(bins):
		begin = int(exp(i*l))
		end = int(exp((i+1)*l))
		if end <= begin: end=begin+1

		result[:, i] = np.sum(a[:, begin:end], axis=1) / (end-begin)
	
	return result

def log_avg2(a, bins):
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
	def __init__(self, bins, fftsize):
		self.borders = np.logspace(0, np.log10(fftsize)*int(bins)/bins, int(bins)+1).astype(int)
		self.borders_end = np.maximum(self.borders[1:], self.borders[:-1]+1)
		self.matrix = np.zeros((fftsize, int(bins)))

		for i in range(int(bins)):
			begin = self.borders[i]
			end = self.borders_end[i]
			for j in range(begin, end):
				self.matrix[j, i] = 1/(end-begin)
		self.matrix = sparse.csr_array(self.matrix) # doesn't really matter if csr or csc

	def log_sum2(self, a):
		return a @ self.matrix

def overlapping_windows(data, window, step):
	if len(data) < len(window):
		return np.zeros((0, len(window)))
	N = len(window)
	step=int(step)
	return np.lib.stride_tricks.as_strided(data,( (len(data)-len(window)) // step + 1, N), [data.strides[0]*step, data.strides[0]]) * window

if args.file == 'jack':
	client = jack.Client('beatdetect')
	audio_in = client.inports.register('audio_in')
	click_out = client.outports.register('click_out')

	total_samples = 0
	last_greedy_beat = 0

	upcoming_beats = []
	current_beats = []

	ringbuf_in = jack.RingBuffer(32000*4)
	ringbuf_out = jack.RingBuffer(32000*4)

	lockout = False

	@client.set_process_callback
	def process(frames):
		global ringbuf_in
		global ringbuf_out
		global lockout

		if frames == 0: return
		
		n = ringbuf_in.write(client.inports[0].get_buffer())
		assert n == frames * 4

		outdata = ringbuf_out.read(frames*4)
		assert len(outdata) == frames*4
		client.outports[0].get_buffer()[:] = outdata



	samplerate = client.samplerate
	bd = BeatDetector(samplerate, force_bpm = args.bpm, timestep_desired_ms = args.timestep)

	with client:
		ringbuf_out.write([0]*8192*4*2)
		while True:
			while ringbuf_in.read_space < 8192*4:
				pass

			data = ringbuf_in.read(8192*4)
			assert len(data) == 8192*4
			data = np.frombuffer(data, dtype=np.float32)

			frames = len(data)
			
			now = total_samples / samplerate
			now_frames = total_samples + 8192*2
			total_samples += frames
			#assert all([b <= now for b in current_beats])

			#print(f"t = {total_samples/samplerate:5.1f}, got {frames} frames, data len is {len(data)}")

			bd.process(data)
			# FIXME this drops beats if there are more than one beat in the time frame.
			if len(bd.greedy_beats) > 0:
				if bd.greedy_beats[-1][0] != last_greedy_beat:
					last_greedy_beat = bd.greedy_beats[-1][0]
					beat_s = bd.greedy_beats[-1][0] * bd.timestep_real
					#print(f"beat at {bd.greedy_beats[-1][0]:7.1f}, {beat_s:5.1f}s, now = {now:5.1f}s, lag = {now-beat_s:6.3}s, frames = {frames}, audio_backlog_pos = {audio_backlog_pos}")

					upcoming_beats = [(bd.greedy_beats[-1][0] + i * bd.greedy_beats[-1][2]) * bd.timestep_real * samplerate for i in range(20)]
					upcoming_beats = [b for b in upcoming_beats if b >= now_frames and all([b >= cb + 0.7*bd.greedy_beats[-1][2] for cb in current_beats])]
					#print(current_beats, upcoming_beats)

			CLICK_FRAMES=int(0.06 * 48000)
			current_beats = [b for b in current_beats if b >= now_frames - CLICK_FRAMES]
			current_beats += [b for b in upcoming_beats if b <= now_frames+frames]
			upcoming_beats = [b for b in upcoming_beats if b >  now_frames+frames]
			#print (now_frames, current_beats, upcoming_beats)
			
			SINE_PERIOD_FRAMES = int(48000//880)
			total_samples=int(total_samples)
			clicks = np.zeros(frames, dtype=np.float32)
			clicks[:] = (np.sin(np.arange(total_samples%SINE_PERIOD_FRAMES, total_samples % SINE_PERIOD_FRAMES +frames) /SINE_PERIOD_FRAMES*2*3.141592654))

			click_mask = np.zeros(frames)
			for b in current_beats:
				b=int(b)
				click_mask[max(0,b-now_frames) : b-now_frames+CLICK_FRAMES] += 1

			clicks[:] *= click_mask
			
			bs = clicks.tobytes()
			n = ringbuf_out.write(bs)
			assert n==len(bs)



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




data = data_orig[:,0]
print(data.shape)


bd = BeatDetector(samplerate, plot_seconds = data.shape[0] / samplerate if args.plot else None, force_bpm = args.bpm, timestep_desired_ms = args.timestep)

for i in range(math.ceil(40 * samplerate / args.chunksize)):
	bd.process(data[i*args.chunksize:args.chunksize*(i+1)])
	if args.plot: plt.show(block=False)



if args.plot and False: # FIXME
	trackerax.clear()
	trackerax.set_xlim(0, args.duration)
	trackerax.set_ylim(-0.05, 1.05)
	trackerax2.clear()
	trackerax2.set_xlim(0, args.duration)
	trackerax2.set_ylim(-0.15, 1.15)

	trackerax2.scatter([b[0]*timestep_real for b in greedy_beats], [b[-1] for b in greedy_beats], color='green')
	trackerax2.scatter([b[0]*timestep_real for b in greedy_beats], [1.07]*len(greedy_beats), color='green')

	scatter_xs = []
	scatter_ys = []
	for t in trackers:
		scatter_xs += [b[0]*timestep_real for b in t.beats]
		scatter_ys += [t.confidence] * len(t.beats)
	trackerax.scatter(scatter_xs, scatter_ys, color='red')

for t in bd.trackers:
	mbt = (t.beats[-1][0] - t.beats[0][0]) / (len(t.beats)-1)
	mbpm = (60/mbt/bd.timestep_real)
	print("tracker suggests %.2f bpm" % mbpm)

beats=np.array( bd.trackers[0].beats ) # FIXME proper getter
greedy_beats = bd.greedy_beats
timestep_real = bd.timestep_real

print("%.2f%%" % (len([1 for b in beats if b[1]]) / len(beats)*100))


mean_beat_time = (beats[-1][0] - beats[0][0]) / (len(beats)-1)

if args.plot:
	axs = bd.axs
	bd.draw_plot()
	for t,f,tpb,prom_next,prom in beats:
		axs[2].axhline(t*timestep_real, color='red', ls='-' if f else '-.')
		axs[2].scatter([prom_next,prom_next * MIN_REL_PROMINENCE], [(t+tpb)*timestep_real]*2, color='blue')
		spanwidth = SIGMA_MS/1000*2
		axs[2].axhspan((t+tpb)*timestep_real-spanwidth/2, (t+tpb)*timestep_real+spanwidth/2, color=("yellow", 0.2))
		axs[2].scatter([prom], [t*timestep_real], color='red')

	for i in range(31):
		#axs[2].axhline(beats[0][0] + i*mean_beat_time, color="green", ls='--')
		#axs[2].axhline((phase + i*periodicity)*timestep_real, color="purple", ls='--')
		pass

mean_bpm = (60/mean_beat_time/timestep_real)
print(f"actual mean tempo = {mean_bpm:.1f}")

errors = np.abs(([b[0] for b in beats] - (beats[0][0] + np.arange(len(beats)) * mean_beat_time)) * timestep_real)
errors_ms = errors*1000

print(f"beat errors: mean = {np.mean(errors_ms):.1f}ms, median = {np.median(errors_ms):.1f}ms, q90 = {np.quantile(errors_ms, 0.9):.1f}ms, max = {np.max(errors_ms):.1f}ms")
#print(f"lambda = {lam}")

if args.plot:
	fig, axs = plt.subplots(1,1)
	ax=axs

	bpms = []
	ts = []
	for (b1, b2) in zip(beats, beats[1:]):
		t1 = b1[0] * timestep_real
		t2 = b2[0] * timestep_real
		bpm = 60 / (t2-t1)
		ts.append((t1+t2)/2)
		bpms.append(bpm)

	ax.plot(ts, bpms)

tt.print_stats()

print("writing out.mp3")

def write_debugout(filename, data_orig, beats):
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
		#if i%2 == 1 and t > 40_000: continue
		t = beat[0]
		f = beat[1]
		beep = beep1 if f else beep2
		t = int((t * timestep_real + time_fixup_s) * samplerate)
		if t < 0: continue
		if t + len(beep) >= data_debug.shape[0]: continue
		data_debug[t:(t+len(beep)), :] += beep.reshape(-1, 1)

	data_debug /= (1 + max(beep))

	sf.write(filename, data_debug, samplerate)

def write_beats(beats, filename):
	with open(filename, 'w') as f:
		for b in beats:
			f.write("%d\n" % b[0])

write_debugout("out.mp3", data_orig, beats)
write_debugout("out_greedy.mp3", data_orig, greedy_beats)

write_beats(beats, "out.txt")
write_beats(greedy_beats, "out_greedy.txt")

if args.plot:
	plt.show()

print("bye")
