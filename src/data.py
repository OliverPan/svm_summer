import numpy as np


class Data(object):
	def __init__(self, data, label):
		self.data = np.array(data, dtype=float)
		self.label = label
		self.predict = None
		self.e = 0
	
	def refresh_e(self):
		self.e = self.predict - self.label
