import numpy as np
import random
import math
import matplotlib.pyplot as plt
from clock_generator import ClockGenerator

class ClockGenerator2:
	def __init__(self, n_ticks, min_delta = 0.1):
		self.n_ticks = n_ticks
		self.beat_time = 0
		self.beat_delta = 1
		self.last_emitted_tick = n_ticks - 1
		self.last_emitted_time = -1
		self.min_delta = min_delta

		self.last_time_to = None

	def update_beats(self, time, delta):
		self.beat_time = time
		self.beat_delta = delta

	def get_ticks(self, time_from, time_to):
		assert self.last_time_to is None or time_from == self.last_time_to
		self.last_time_to = time_to

		result = []

		next_beat_index = int(math.ceil((time_from - self.beat_time) / self.beat_delta)) # smallest x s.t. self.beat_time + x*self.beat_delta >= time_from
		next_beat_time = self.beat_time + next_beat_index * self.beat_delta

		print(f'\n\nget_ticks {time_from} .. {time_to}, {next_beat_index=} {next_beat_time=}\n')
		assert next_beat_time >= time_from
		assert self.last_emitted_time < time_from

		# if we're clean (last beat has been fully emitted) but we just missed the next beat, let's just catch up instead of waiting almost a full beat
		if self.last_emitted_tick == self.n_ticks - 1 and next_beat_time - (self.last_emitted_time + self.beat_delta/self.n_ticks) >= self.beat_delta / 2:
			print(f"we're late, catching up at {time_from}")
			result += [(time_from, 0)]
			self.last_emitted_tick = 0
			self.last_emitted_time = time_from

		# if we're not clean: emit the remaining ticks to finish the last beats, such that the next beat can happen in time
		if self.last_emitted_tick != self.n_ticks - 1:
			# remaining ticks and time until (including) the next "0" tick
			remaining_ticks = self.n_ticks - self.last_emitted_tick
			remaining_time = next_beat_time - self.last_emitted_time
			temporary_delta = remaining_time / remaining_ticks
			print(f"not clean at tick {self.last_emitted_tick}, {remaining_ticks=}, {remaining_time=}, {temporary_delta=}")

			if temporary_delta < self.min_delta:
				remaining_ticks += self.n_ticks
				remaining_time += self.beat_delta
				temporary_delta = remaining_time / remaining_ticks
				next_beat_index += 1
				next_beat_time += self.beat_delta
				print(f"delta too small 1, new {temporary_delta=}, {next_beat_time=}")

			if self.last_emitted_time + temporary_delta < time_from:
				print(f"catch up would be in the past ({self.last_emitted_time + temporary_delta} < {time_from})), realigning")
				# special case: the first "catch-up" tick would be in the past -> start the catch-up ticks at the earliest possible time and squish them together as needed
				remaining_ticks -= 1
				remaining_time = next_beat_time - time_from
				temporary_delta = remaining_time / remaining_ticks
				print(f"  new {temporary_delta=}")

				if temporary_delta < self.min_delta:
					print(f"delta too small 2")
					remaining_ticks += 0#self.n_ticks
					remaining_time += self.beat_delta
					temporary_delta = remaining_time / remaining_ticks
					next_beat_index += 1
					next_beat_time += self.beat_delta
					print(f"  new {temporary_delta=} for {remaining_ticks=}, new {next_beat_index=} {next_beat_time=}")

				first = time_from
				remaining_ticks +=1 # DIRTY HACK FIXME
			else:
				print(f"catch up is fine")
				first = self.last_emitted_time + temporary_delta

			remaining_ticks -= 1 # we won't generate the final (i.e. #0) tick, because the code below will create it.
			remaining_ticks = min(remaining_ticks, int(math.ceil((time_to - first) / temporary_delta - 1e-4))) # limit to the largest x s.t. first + (x-1)*temporary_delta < time_to. equiv to smallest x s.t. first+x*tempdelt >= time_to
			print(f"catching up from {first} to {time_to} with {temporary_delta=}. {remaining_ticks=}")
			
			result += [(first + i*temporary_delta, (i+self.last_emitted_tick+1) % self.n_ticks) for i in range(remaining_ticks)]
			self.last_emitted_time = first + (remaining_ticks-1)*temporary_delta
			self.last_emitted_tick = (self.last_emitted_tick+remaining_ticks) % self.n_ticks
			print (remaining_ticks, first + (remaining_ticks-1)*temporary_delta, time_to)
			assert self.last_emitted_time < time_to
		
		if next_beat_time < time_to:
			print("emitting beat")
			print (self.last_emitted_tick)
			assert self.last_emitted_time < next_beat_time
			assert self.last_emitted_tick == self.n_ticks-1
			# now we're clean to start at tick 0
			tick_delta = self.beat_delta / self.n_ticks
			gen_ticks = int(math.ceil((time_to - next_beat_time) / tick_delta)) # largest x s.t. next_beat_time + (x-1)*tick_delta < time_to; equiv to smallest x s.t. next_beat_time + x*tick_delta >= time_to
			result += [(next_beat_time + i * tick_delta, i%self.n_ticks) for i in range(gen_ticks)]

			self.last_emitted_time = next_beat_time + (gen_ticks-1) * tick_delta
			self.last_emitted_tick = (gen_ticks-1) % self.n_ticks

		return result

def plot(ticks, n_ticks, c=None):
	if c is not None:
		plt.plot([t for t,i in ticks], np.unwrap([i for t,i in ticks], period = n_ticks), linestyle='-', marker='o', color=c)
	else:
		plt.scatter([t for t,i in ticks], np.unwrap([i for t,i in ticks], period = n_ticks), c=[i for t,i in ticks])

c = ClockGenerator(12)

ticks = []

c.update_beats(0, 24)
r = c.get_ticks(0, 48)
print(r)
ticks += r

r = c.get_ticks(48, 72)
print(r)
ticks += r

c.update_beats(60, 12)
r = c.get_ticks(72, 90)
print(r)
ticks += r

c.update_beats(84, 24)
r= c.get_ticks(90, 132)
print(r)
ticks += r

c.update_beats(131, 24)
r = c.get_ticks(132, 180)
print(r)
ticks += r

def sanitycheck(ticks, n_ticks):
	last_time = -9999
	for i, (time, tick) in enumerate(ticks):
		assert time > last_time
		assert tick == i % n_ticks
		last_time = time

def test_aligned(updates, n):
	results = [[] for i in range(n)]
	cs = [ClockGenerator(12) for i in range(n)]

	for c in cs:
		c.update_beats(updates[0][1], updates[0][2])

	for (ut0, bt0, tpb0), (ut1, bt1, tpb1) in zip(updates, updates[1:]):
		for i, (result, c) in enumerate(zip(results, cs)):
			n_chunks = i if i > 0 else 1# (ut1-ut0)
			cuts = np.linspace(ut0, ut1, n_chunks+1)
			for cut0, cut1 in zip(cuts, cuts[1:]):
				print(f"\n\n##### processing {i} from {cut0} to {cut1}\n")
				result += c.get_ticks(cut0, cut1)
			c.update_beats(bt1, tpb1)
	
	for i, (result, c) in enumerate(zip(results, cs)):
		print(f"checking {i+1}")
		sanitycheck(result, 12)

		assert len(result) == len(results[0])
		for (ta, ia), (tb, ib) in zip(result,results[0]):
			assert ia==ib
			assert abs(ta-tb) < 1e-5

		plot(result, 12, f"C{i}")

	for (t, _, _) in updates:
		plt.axvline(t)

	for i in range(0,140, 12):
		plt.axhline(i)

	plt.legend([f'C{i}' for i in range(n)])
	plt.show()

def stresstest():
	tempo = random.normalvariate(140, 30)

	t = 48000
	updates=[]
	for i in range(0, 48000*10, 48000*2):
		updates+=[(t,t-random.uniform(1, 0.3*48000),48000*60/tempo)]
		tempo += random.normalvariate(0,1)
		t+=48000*2
		print(i, tempo)
	test_aligned(updates,12)
stresstest()


my_updates = [
	(  0,  0, 24),
	( 48, 48, 12),
	( 72, 72, 24),
	( 87, 84, 20),
	(105,103, 20),
	(140,129, 22),
	(180,157, 23),
	(200,182, 21),
	(220,219, 19),
]

my_updates = [
	(140-140,129-140, 22),
	(180-140,157-140, 23),
	(200-140,182-140, 21),
	(220-140,219-140, 19),
]

test_aligned(my_updates, 10)
	

plot(ticks, 12)
plt.show()


