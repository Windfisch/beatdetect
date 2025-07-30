import time
import numpy as np

class TimeTracker:
	def __init__(self):
		self.ordered_keys = []
		self.times = {}
		self.last_timestamp = None
		self.last_name = None

	def begin(self, name):
		if self.last_name is not None:
			elapsed = time.time() - self.last_timestamp
			exists = self.last_name in self.times
			if not exists:
				self.times[self.last_name] = []
				self.ordered_keys.append(self.last_name)

			self.times[self.last_name].append(elapsed)

		self.last_name = name
		self.last_timestamp = time.time()
	
	def print_stats(self):
		print(f"{'':50s}  {'total':>9s} ={'number':>8s} x {'average':>10s} ± {'stddev':>10s}")
		for key in self.ordered_keys:
			print(f"{key:50s}: {np.sum(self.times[key]):8.2f}s ={len(self.times[key]):8d} x {1000*np.average(self.times[key]):8.1f}ms ± {1000*np.std(self.times[key]):8.1f}ms")

		total = sum(sum(self.times.values(), start=[]))
		print(f"{'TOTAL':50s}: {total:8.2f}s")

