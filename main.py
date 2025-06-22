import soundfile as sf
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from numpy import log, exp

def cfar_kernel(n, guard):
	#return np.array( [-0.5/n]*n + [0]*guard + [1] + [0]*guard + [-0.5/n]*n )
	return np.array( [0]*n + [0]*guard + [1] + [0]*guard + [-1/n]*n )
	#return np.array( [-1/n]*n + [0]*guard + [1] + [0]*guard + [0]*n )


print("reading file")
#data, samplerate = sf.read("vimana.mp3")
#t0=100
data, samplerate = sf.read(sys.argv[1])
t0=int(sys.argv[2])

print(len(data))
print(data[1])
print(samplerate)

print("numpy-fying")
data = np.array(data)

print(data.shape)

data = data[:,0]
print(data.shape)



print("trimming")
data = data[t0*samplerate: (t0+5)*samplerate]
#data = data[120*samplerate: 130*samplerate]

timestep_desired = 1/1000
samplestep = int(samplerate * timestep_desired)
timestep_real = samplestep / samplerate

print(f"using a timestep of {timestep_real*1000:.3f}ms")

def overlapping_windows(data, window, step):
	N = len(window)
	step=int(step)
	print(N, step, data.strides)
	print(len(data), len(window))
	return np.lib.stride_tricks.as_strided(data,( (len(data)-len(window)) // step ,N), [data.strides[0]*step, data.strides[0]]) * window

x = overlapping_windows(data, np.hamming(samplerate/20), samplestep)

y = np.absolute( np.fft.rfft(x, axis=1) )

y = np.log10(y + 1e-4) * 20

#y = ss.convolve2d(y, np.transpose([cfar_kernel( int(samplerate*0.01), int(samplerate*0.01) )]))
print("convolving")
#my_kernel = cfar_kernel( int(samplerate*0.1), int(samplerate*0.1) )

my_kernel = cfar_kernel( int(0.2 / timestep_real), int(0.04 / timestep_real) )
y = np.apply_along_axis( lambda foo: ss.oaconvolve(foo, my_kernel, 'valid'), axis=0, arr=y )

y = np.fmax(y - 10, 0)


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
		
bins = math.log(y.shape[1], 2)*12
y = log_avg2(y, bins)

#y=np.apply_along_axis( lambda foo: find_local_maxima(foo), axis=0, arr=y) & (y>0)
print("done")

#y = np.apply_along_axis( lambda foo: ss.correlate(foo, foo[0:int(1/timestep_real)], 'valid'), axis=0, arr=y)

#plt.xscale('log')
#plt.imshow(y, aspect='auto', vmax=1e+4, vmin=0)

print("quantile")

#vmax = np.quantile(y, 0.999, method='inverted_cdf')
#vmax2 = np.quantile(y, 0.99, method='inverted_cdf')
#print(vmax, vmax2)
vmin = 0

print("imshow")
#plt.imshow(y, aspect='auto')

z = np.sum( y[:, 50:], axis=1)

plt.plot(z)

#t=np.linspace(0,y.shape[0], y.shape[0])
#f=np.linspace(0,y.shape[1], y.shape[1])

#plt.y

plt.show()
