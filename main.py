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
p.add_argument('--early-binning', action='store_true')
p.add_argument('--bpm')
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

data, samplerate = sf.read(args.file)
t0=int(args.start)

print(len(data))
print(data[1])
print(samplerate)

print("numpy-fying")
data = np.array(data)

print(data.shape)

data = data[:,0]
print(data.shape)



print("trimming")
data = data[t0*samplerate: (t0+15)*samplerate]

timestep_desired = 1/1000
samplestep = int(samplerate * timestep_desired)
timestep_real = samplestep / samplerate

print(f"using a timestep of {timestep_real*1000:.3f}ms")
print("windowing")

x = overlapping_windows(data, np.hamming(samplerate/40), samplestep)

fig, axs = plt.subplots(2,3)

axs = np.ndarray.flatten(axs)

print("fft")
y = np.absolute( np.fft.rfft(x, n = 2*x.shape[1], axis=1) )

if args.early_binning:
	print("binning")
	bins = math.log(y.shape[1], 2)*12
	y = log_sum2(y, bins)

y = np.log10(y + 1e-4) * 20

if not args.early_binning:
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
y = np.fmax(y - 10, 0)

if not args.early_binning:
	print("binning")
	bins = math.log(y.shape[1], 2)*12
	y = log_avg2(y, bins)

axs[1].imshow(y, aspect='auto') #, vmax=1e+4, vmin=0)
if args.early_binning:
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

t = np.argmax(phases)
phase = np.argmax(phases)
delta_t = periodicity

half_window = int(periodicity/2 * 0.7)

beats = []
for i in range(31):
	t += delta_t
	if t < half_window:
		continue
	if t+half_window > len(z): break

	window = z[t-half_window : t+half_window+1]

	result = ss.find_peaks(window, height=0, distance = (0.01 / timestep_real), prominence=0.05 * np.max(window))

	indices = result[1]['peak_heights'].argsort()[::-1][0:5]

	peaks = result[0][indices] + (t-half_window)
	heights = result[1]['peak_heights'][indices]


	print(f"peaks = {peaks}, heights = {heights}")
	#print(result[1])

	if len(peaks) == 0: continue
	t = peaks[0]


	beats.append(t)
	axs[2].axhline(t, color='red')

beats=np.array(beats)

mean_beat_time = (beats[-1] - beats[0]) / (len(beats)-1)

for i in range(31):
	axs[2].axhline(beats[0] + i*mean_beat_time, color="green", ls='--')
	axs[2].axhline(phase + i*periodicity, color="purple", ls='--')

mean_bpm = (60/mean_beat_time/timestep_real)
print(f"original tempo estimate = {tempo:.1f}bpm, actual mean tempo = {mean_bpm:.1f}")

errors = np.abs((beats - (beats[0] + np.arange(len(beats)) * mean_beat_time)) * timestep_real)
errors_ms = errors*1000

print(f"beat errors: mean = {np.mean(errors_ms):.1f}ms, median = {np.median(errors_ms):.1f}ms, q90 = {np.quantile(errors_ms, 0.9):.1f}ms, max = {np.max(errors_ms):.1f}ms")


#t=np.linspace(0,y.shape[0], y.shape[0])
#f=np.linspace(0,y.shape[1], y.shape[1])

#plt.y

plt.show()
