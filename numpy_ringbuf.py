import numpy as np

class Ringbuf2D:
	def __init__(self, size, width, oversize_factor=2):
		self.size = size
		self.buffer = np.zeros((size * oversize_factor, width))
		self.pos = 0
	
	def append(self, data):
		if data.shape[1] != self.buffer.shape[1]:
			raise ValueError(f"cannot add data with width={data.shape[1]} to ringbuffer of width={self.buffer.shape[1]}")

		if data.shape[0] > self.size:
			data = data[-self.size: , :]

		if self.pos + data.shape[0] > self.buffer.shape[0]:
			keep = self.size - data.shape[0]
			self.buffer[0:keep, :] = self.buffer[self.pos - keep : self.pos, :]
			self.pos = keep

		self.buffer[self.pos : self.pos+data.shape[0], :] = data
		self.pos += data.shape[0]
	
	def get(self):
		return self.buffer[max(0, self.pos - self.size):self.pos, :]


class Ringbuf1D:
	def __init__(self, size, oversize_factor=2):
		self.size = size
		self.buffer = np.zeros(size * oversize_factor)
		self.pos = 0
	
	def append(self, data):
		if len(data) > self.size:
			data = data[-self.size:]

		if self.pos + len(data) > len(self.buffer):
			keep = self.size - len(data)
			self.buffer[0:keep] = self.buffer[self.pos - keep : self.pos]
			self.pos = keep

		self.buffer[self.pos : self.pos+len(data)] = data
		self.pos += len(data)
	
	def get(self):
		return self.buffer[max(0, self.pos - self.size):self.pos]

