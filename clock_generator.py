import math

class ClockGenerator:
	def __init__(self, n_ticks, min_delta = 0.1, verbose=False):
		self.n_ticks = n_ticks
		self.beat_time = 0
		self.beat_delta = 1
		self.last_emitted_tick = n_ticks - 1
		self.last_emitted_time = -1
		self.min_delta = min_delta
		self.verbose = verbose

		self.last_time_to = None

	def update_beats(self, time, delta):
		self.beat_time = time
		self.beat_delta = delta

	def get_ticks(self, time_from, time_to):
		result = []
		self.get_ticks_cb(time_from, time_to, lambda time, tick_index : result.append((time, tick_index)))
		return result

	def get_ticks_cb(self, time_from, time_to, tick_func):
		assert self.last_time_to is None or time_from == self.last_time_to
		self.last_time_to = time_to

		next_beat_index = int(math.ceil((time_from - self.beat_time) / self.beat_delta)) # smallest x s.t. self.beat_time + x*self.beat_delta >= time_from
		next_beat_time = self.beat_time + next_beat_index * self.beat_delta

		if self.verbose: print(f'\n\nget_ticks {time_from} .. {time_to}, {next_beat_index=} {next_beat_time=}\n')
		assert next_beat_time >= time_from
		assert self.last_emitted_time < time_from

		# if we're clean (last beat has been fully emitted) but we just missed the next beat, let's just catch up instead of waiting almost a full beat
		if self.last_emitted_tick == self.n_ticks - 1 and next_beat_time - (self.last_emitted_time + self.beat_delta/self.n_ticks) >= self.beat_delta / 2:
			if self.verbose: print(f"we're late, catching up at {time_from}")
			tick_func(time_from, 0)
			self.last_emitted_tick = 0
			self.last_emitted_time = time_from

		# if we're not clean: emit the remaining ticks to finish the last beats, such that the next beat can happen in time
		if self.last_emitted_tick != self.n_ticks - 1:
			# remaining ticks and time until (including) the next "0" tick
			remaining_ticks = self.n_ticks - self.last_emitted_tick
			remaining_time = next_beat_time - self.last_emitted_time
			temporary_delta = remaining_time / remaining_ticks
			if self.verbose: print(f"not clean at tick {self.last_emitted_tick}, {remaining_ticks=}, {remaining_time=}, {temporary_delta=}")

			if temporary_delta < self.min_delta:
				remaining_ticks += self.n_ticks
				remaining_time += self.beat_delta
				temporary_delta = remaining_time / remaining_ticks
				next_beat_index += 1
				next_beat_time += self.beat_delta
				if self.verbose: print(f"delta too small 1, new {temporary_delta=}, {next_beat_time=}")

			if self.last_emitted_time + temporary_delta < time_from:
				if self.verbose: print(f"catch up would be in the past ({self.last_emitted_time + temporary_delta} < {time_from})), realigning")
				# special case: the first "catch-up" tick would be in the past -> start the catch-up ticks at the earliest possible time and squish them together as needed
				remaining_ticks -= 1
				remaining_time = next_beat_time - time_from
				temporary_delta = remaining_time / remaining_ticks
				if self.verbose: print(f"  new {temporary_delta=}")

				if temporary_delta < self.min_delta:
					if self.verbose: print(f"delta too small 2")
					remaining_ticks += 0#self.n_ticks
					remaining_time += self.beat_delta
					temporary_delta = remaining_time / remaining_ticks
					next_beat_index += 1
					next_beat_time += self.beat_delta
					if self.verbose: print(f"  new {temporary_delta=} for {remaining_ticks=}, new {next_beat_index=} {next_beat_time=}")

				first = time_from
				remaining_ticks +=1 # DIRTY HACK FIXME
			else:
				if self.verbose: print(f"catch up is fine")
				first = self.last_emitted_time + temporary_delta

			remaining_ticks -= 1 # we won't generate the final (i.e. #0) tick, because the code below will create it.
			remaining_ticks = min(remaining_ticks, int(math.ceil((time_to - first) / temporary_delta - 1e-4))) # limit to the largest x s.t. first + (x-1)*temporary_delta < time_to. equiv to smallest x s.t. first+x*tempdelt >= time_to
			if self.verbose: print(f"catching up from {first} to {time_to} with {temporary_delta=}. {remaining_ticks=}")
			
			for i in range(remaining_ticks):
				tick_func(first + i*temporary_delta, (i+self.last_emitted_tick+1) % self.n_ticks)

			self.last_emitted_time = first + (remaining_ticks-1)*temporary_delta
			self.last_emitted_tick = (self.last_emitted_tick+remaining_ticks) % self.n_ticks
			if self.verbose: print (remaining_ticks, first + (remaining_ticks-1)*temporary_delta, time_to)
			assert self.last_emitted_time < time_to
		
		if next_beat_time < time_to:
			if self.verbose: print("emitting beat")
			if self.verbose: print (self.last_emitted_tick)
			assert self.last_emitted_time < next_beat_time
			assert self.last_emitted_tick == self.n_ticks-1
			# now we're clean to start at tick 0
			tick_delta = self.beat_delta / self.n_ticks
			gen_ticks = int(math.ceil((time_to - next_beat_time) / tick_delta)) # largest x s.t. next_beat_time + (x-1)*tick_delta < time_to; equiv to smallest x s.t. next_beat_time + x*tick_delta >= time_to
			for i in range(gen_ticks):
				tick_func(next_beat_time + i * tick_delta, i%self.n_ticks)

			self.last_emitted_time = next_beat_time + (gen_ticks-1) * tick_delta
			self.last_emitted_tick = (gen_ticks-1) % self.n_ticks
