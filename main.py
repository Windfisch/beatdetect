import soundfile as sf
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.ndimage as sn
from numpy import log, exp
import time
import argparse

p = argparse.ArgumentParser()
p.add_argument('file')
p.add_argument('start')
p.add_argument('--late-binning', action='store_true')
p.add_argument('--bpm')
p.add_argument('--offbeat', action='store_true')
p.add_argument('--duration', type=float, default=20)
args = p.parse_args()

print_orig = print
last_timestamp = None
def print(*args, **kwargs):
	global last_timestamp
	if last_timestamp is None: last_timestamp = time.time()
	diff = time.time() - last_timestamp
	last_timestamp = time.time()
	print_orig(f'[+{diff*1000:7.2f}ms]', *args, **kwargs)

def cfar_kernel(n, guard):
	return np.array( [1] + [0]*guard + [-1/n]*n )

def find_local_maxima(a):
	return (np.convolve(a, [1,-1])[:-1]>=0) & (np.convolve(a, [1,-1])[1:]<=0)

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

def log_sum2(a, bins):
	l = log(a.shape[1]) / bins
	bins = int(bins)
	result=np.zeros((a.shape[0], bins))

	for i in range(bins):
		begin = int(exp(i*l))
		end = int(exp((i+1)*l))
		if end <= begin: end=begin+1

		result[:, i] = np.sum( a[:, begin:end], axis=1) / (end-begin)
	
	return result

def overlapping_windows(data, window, step):
	N = len(window)
	step=int(step)
	print(N, step, data.strides)
	print(len(data), len(window))
	return np.lib.stride_tricks.as_strided(data,( (len(data)-len(window)) // step ,N), [data.strides[0]*step, data.strides[0]]) * window


print("reading file")

data_orig, samplerate = sf.read(args.file)
t0=int(args.start)

print(len(data_orig))
print(data_orig[1])
print(samplerate)

print("numpy-fying")
data_orig = np.array(data_orig)

print("trimming")
data_orig = data_orig[int(t0*samplerate): int((t0+args.duration)*samplerate), :]
print(data_orig.shape)




data = data_orig[:,0]
print(data.shape)




timestep_desired = 1/1000
samplestep = int(samplerate * timestep_desired)
timestep_real = samplestep / samplerate

print(f"using a timestep of {timestep_real*1000:.3f}ms")
print("windowing")

fft_window_ms = 25

x = overlapping_windows(data, np.hamming(int(fft_window_ms / 1000 * samplerate)), samplestep)

fig, axs = plt.subplots(2,3)

axs = np.ndarray.flatten(axs)

print("fft")
y = np.absolute( np.fft.rfft(x, n = 2*x.shape[1], axis=1) )

if not args.late_binning:
	print("binning")
	bins = math.log(y.shape[1], 2)*12
	y = log_sum2(y, bins)

y = np.log10(y + 1e-4) * 20

if args.late_binning:
	axs[0].set_xscale('log')

waterfall = y

#y = ss.convolve2d(y, np.transpose([cfar_kernel( int(samplerate*0.01), int(samplerate*0.01) )]))
print("convolving")
#my_kernel = cfar_kernel( int(samplerate*0.1), int(samplerate*0.1) )

my_kernel = cfar_kernel( int(0.2 / timestep_real), int(0.04 / timestep_real) )
print("kernel len = ", len(my_kernel))
y = np.apply_along_axis( lambda foo: ss.oaconvolve(foo, my_kernel, 'valid'), axis=0, arr=y )

waterfall = waterfall[len(my_kernel):, :]
axs[0].imshow(waterfall, aspect='auto') #, vmax=1e+4, vmin=0)

print("maximum")
y = np.fmax(y - 0, 0)

if args.late_binning:
	print("binning")
	bins = math.log(y.shape[1], 2)*12
	y = log_avg2(y, bins)

axs[1].imshow(y, aspect='auto', vmin=0, vmax=np.quantile(y, 0.97)*0.8)
if not args.late_binning:
	axs[1].sharex(axs[0])
axs[1].sharey(axs[0])

#y=np.apply_along_axis( lambda foo: find_local_maxima(foo), axis=0, arr=y) & (y>0)
print("done")

#y = np.apply_along_axis( lambda foo: ss.correlate(foo, foo[0:int(1/timestep_real)], 'valid'), axis=0, arr=y)

min_bpm = 60
max_bpm = 300
correlation_window = int(20/timestep_real)

#corr_data = y[ -int((60/min_bpm) / timestep_real)-correlation_window:, :]
#corr_reference = corr_data[-correlation_window:, :]
#correlation = np.apply_along_axis( lambda foo: np.flip(ss.correlate(
#	foo[ -int((60/min_bpm) / timestep_real)-correlation_window:],
#	foo[-correlation_window:], 'valid')), axis=0, arr=y)
correlation = np.apply_along_axis( lambda foo: np.flip(ss.correlate(
	foo[ 0 : int((60/min_bpm) / timestep_real)+correlation_window],
	foo[ int((60/min_bpm) / timestep_real) :int((60/min_bpm) / timestep_real)+correlation_window],
	'valid')), axis=0, arr=y)

correlation_1d = np.sum(correlation, axis=1)

crop = int(60/max_bpm/timestep_real)

peak_limit = (np.quantile(correlation_1d[crop:], 0.9))

peaks = ss.find_peaks(correlation_1d, height=peak_limit, prominence=peak_limit*0.3)[0]
tempi = [ (60/(t*timestep_real)) for t in peaks]
print("possible tempi: " + ", ".join(["%.1fbpm" % t for t in tempi]))

if args.bpm is None:
	tempo = tempi[0]
else:
	tempo = float(args.bpm)
	print("Using --bpm override")

if tempo > 240: tempo /= 2

periodicity = int(np.round(60/tempo/timestep_real))




axs[3].imshow(correlation, aspect='auto')
axs[4].set_xlim(xmin=0, xmax=np.max(correlation_1d[crop:]))
axs[4].plot(correlation_1d, np.arange(len(correlation_1d)))
axs[4].axvline(peak_limit)
axs[4].axvline(peak_limit)

if args.bpm is not None:
	axs[4].axhline(periodicity)
	axs[3].axhline(periodicity)

axs[4].sharey(axs[3])

#plt.xscale('log')
#plt.imshow(y, aspect='auto', vmax=1e+4, vmin=0)

print("quantile")

#vmax = np.quantile(y, 0.999, method='inverted_cdf')
#vmax2 = np.quantile(y, 0.99, method='inverted_cdf')
#print(vmax, vmax2)
vmin = 0

print("imshow")

#plt.imshow(y, aspect='auto')

z = np.sum( y[:, 40:], axis=1)

axs[2].plot(z, np.arange(len(z)), color='orange')
axs[2].plot(sn.gaussian_filter1d(z, 10), np.arange(len(z)), color='blue')
z = sn.gaussian_filter1d(z, 10)
axs[2].sharey(axs[0])

for tempo_ in tempi:
	periodicity_ = int(np.round(60/tempo_/timestep_real))
	phase_window = int((5 / timestep_real) / periodicity_)*periodicity_
	n = phase_window / periodicity_
	phases = z[:phase_window].reshape(-1, periodicity_).sum(axis=0) / n
	axs[5].plot(phases, lw=1)

phase_window = int((5 / timestep_real) / periodicity)*periodicity
n = phase_window / periodicity
phases = z[:phase_window].reshape(-1, periodicity).sum(axis=0) / n
axs[5].plot(phases)



#############


# assume peak prominences p are distributed with a density of f(p)
# i.e. P(p <= x) = integral of f(t)dt from -inf to x.
#
# Then if we have a peak with prominence p, what's the probability of this peak being caused by the
# distribution (i.e. a "noisy" one, not a real peak)?
#
# Hypothesis Hi: peak i is the beat
# Hypothesis H-: none of the peaks is the beat
#
# Let's say p_i := (loc_i, prom_i) is the list of our detected peaks and p = (p_1, ..., p_n)
# P( p | Hi ) ~  product(f(prom_k)) / f(prom_i) * g(loc_i)
# thus:
# P( Hi | p ) = P ( p | Hi )

print("Running statistics on the peaks")
#z=z[0:8000]
result = ss.find_peaks(z, height=0, distance = (0.01 / timestep_real), prominence=0)
prominences = sorted(result[1]['prominences'])[::-1]
lam = 1/np.mean(prominences)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(prominences, np.arange(len(prominences)))
ax.plot(np.arange(max(prominences)), len(prominences) * np.exp(- (lam * np.arange(max(prominences)))**0.8 ))
ax.plot(np.arange(max(prominences)), len(prominences) * np.exp(-lam * np.arange(max(prominences)) ))



serial_number = 1

class BeatTracker:
	# both beat_loc and time_per_beat are given in timesteps, i.e. not neccessarily msec
	def __init__(self, parent, sigma, lam, beat_loc, time_per_beat, confidence, beats):
		global serial_number
		self.sigma = sigma
		self.parent = parent
		self.serial = serial_number
		serial_number+=1
		self.lam = lam
		self.beat_loc = beat_loc
		self.time_per_beat = time_per_beat
		self.confidence = confidence
		self.beats = beats[:]
		if confidence > 0.05:
			print(f"Tracker #{self.parent} spawns #{self.serial} at {self.beat_loc} with {confidence*100:.2f}%, expected = {self.time_per_beat+self.beat_loc}")

	def search_interval(self):
		expected_loc = self.beat_loc + self.time_per_beat
		return (expected_loc - 300, expected_loc+300)

	def next_beat(self, peak_locs, peak_prominences):
		expected_loc = self.beat_loc + self.time_per_beat
		peaks = []

		for loc, prom in zip(peak_locs, peak_prominences):
			t_diff = loc - expected_loc
			relevance = math.exp(-0.5*(t_diff/self.sigma)**2) / (self.lam * math.exp(-self.lam * prom))
			peaks.append((relevance, loc, True))

		P_HAVEPEAK = 0.03
		peaks.append((1/P_HAVEPEAK-1, expected_loc, False))

		relevance_sum = sum([r for r,l,f in peaks])
		peaks = [(r/relevance_sum, l, f) for r,l,f in peaks]
		peaks = sorted(peaks)[::-1]

		alpha = 0.1
		confidence = self.confidence * (1-alpha) + 1 * alpha

		return [BeatTracker(self.serial, self.sigma, self.lam, loc, self.time_per_beat, confidence * rel, self.beats + [(loc, found)]) for (rel, loc, found) in peaks]

def get_search_interval(trackers):
	lo = 999999
	hi = -1
	for t in trackers:
		l, h = t.search_interval()
		lo = min(lo, l)
		hi = max(hi, h)
	return (lo, hi)

t = np.argmax(phases)

if args.offbeat:
	print("offbeat foo")
	t += int(periodicity/2)

phase = np.argmax(phases)
delta_t = periodicity

trackers = [BeatTracker(0, 20, lam, t, periodicity, 1, [(t, False)])]

half_window = int(periodicity/2 * 0.7)

beats = []
for i in range(99):
	window_start, window_end = get_search_interval(trackers)
	window_start = min(max(window_start, 0), len(z))
	window_end = min(max(window_end, 0), len(z))
	print(f"window = {window_start}..{window_end}, len = {window_end-window_start}")
	window = z[window_start : window_end]

	if len(window) < 10: break

	result = ss.find_peaks(window, height=0, distance = (0.01 / timestep_real), prominence=0.05 * np.max(window))

	peaks = result[0] + window_start
	heights = result[1]['peak_heights']

	#print(f"peaks = {peaks}, heights = {heights}")
	print(f"found {len(peaks)} peaks")

	if len(peaks) == 0: continue

	trackers_new = []
	for tr in trackers:
		trackers_new += tr.next_beat(peaks, heights)

	print("------------------------------------")

	trackers = trackers_new

	trackers.sort(key = lambda t : -t.confidence)

	trackers = trackers[0:10]
	print("confidences: ", ", ".join(["%5f" % t.confidence for t in trackers]))

for t in trackers:
	mbt = (t.beats[-1][0] - t.beats[0][0]) / (len(t.beats)-1)
	mbpm = (60/mbt/timestep_real)
	print("tracker suggests %.2f bpm" % mbpm)

beats=np.array( trackers[0].beats )

for t,f in beats:
	axs[2].axhline(t, color='red', ls='-' if f else '-.')

print("%.2f%%" % (len([1 for t,f in beats if f]) / len(beats)*100))


mean_beat_time = (beats[-1][0] - beats[0][0]) / (len(beats)-1)

for i in range(31):
	#axs[2].axhline(beats[0][0] + i*mean_beat_time, color="green", ls='--')
	axs[2].axhline(phase + i*periodicity, color="purple", ls='--')

mean_bpm = (60/mean_beat_time/timestep_real)
print(f"original tempo estimate = {tempo:.1f}bpm, actual mean tempo = {mean_bpm:.1f}")

errors = np.abs(([b for b,f in beats] - (beats[0][0] + np.arange(len(beats)) * mean_beat_time)) * timestep_real)
errors_ms = errors*1000

print(f"beat errors: mean = {np.mean(errors_ms):.1f}ms, median = {np.median(errors_ms):.1f}ms, q90 = {np.quantile(errors_ms, 0.9):.1f}ms, max = {np.max(errors_ms):.1f}ms")
print(f"lambda = {lam}")

print("writing out.mp3")

data_debug = data_orig[:,:]

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
beep1 = beep1 * beep_window * 0.5
beep2 = beep2 * beep_window * 0.5


for i, (t,f) in enumerate(beats):
	beep = beep1 if f else beep2
	t = int((t * timestep_real + fft_window_ms/1000/2) * samplerate)
	if t < 0: continue
	if t + len(beep) >= data_debug.shape[0]: continue
	data_debug[t:(t+len(beep)), :] += beep.reshape(-1, 1)

data_debug /= (1 + max(beep))

sf.write("out.mp3", data_debug, samplerate)
#t=np.linspace(0,y.shape[0], y.shape[0])
#f=np.linspace(0,y.shape[1], y.shape[1])

#plt.y

plt.show()
