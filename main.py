from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import soundfile as sf
import gc
import math
import numpy as np
import argparse


from timetracker import TimeTracker
from data import Beat, GreedyBeat

from beatdetector import BeatDetector, SIGMA_MS
from beattracker import MIN_REL_PROMINENCE # FIXME
from live import run_live


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
p.add_argument('--miditap', type=str, default='0x54:1', help='note:channel; note can be 123 or 0x45; channel is 1-based-indexed.')
p.add_argument('--plot', action='store_true', default=False)
p.add_argument('--timestep', type=float, default=1)
p.add_argument('--chunksize', type=int, default=-1)
args = p.parse_args()

if args.plot:
	import matplotlib.pyplot as plt
	import matplotlib

tt = TimeTracker()


if args.file == 'jack':
	miditap_note = None
	miditap_channel = None
	if args.miditap is not None:
		n, c = args.miditap.split(':')
		miditap_note = int(n,0)
		miditap_channel = int(c,0) if c != '*' else 0
		if miditap_channel == 0:
			miditap_channel = None
		else:
			miditap_channel -= 1
	run_live(args.timestep, args.bpm, not args.nogui, miditap_note, miditap_channel)
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
