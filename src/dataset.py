import matplotlib.pyplot as plt
import numpy as np
import random
import data


class Dataset:
	def __init__(self, dimension, key="liner"):
		self.dimension = dimension
		self.set = []
		self.key = key
		
	def init_set(self, num):
		if self.key == "liner":
			time = 0
			while time < num:
				x = random.uniform(0, 10)
				y = random.uniform(0, 10)
				if abs(x-y) >= 1:
					if x-y >= 1:
						self.set.append(data.Data([x, y], 1))
					else:
						self.set.append(data.Data([x, y], -1))
					time += 1
	
	def show(self):
		for node in self.set:
			if float(10/7) > node[0] > 0:
				plt.scatter(node[0], node[1], color="red")
			elif float(20/7) > node[0] > float(10/7):
				plt.scatter(node[0], node[1], color="orange")
			elif float(30/7) > node[0] > float(20/7):
				plt.scatter(node[0], node[1], color="yellow")
			elif float(40/7) > node[0] > float(30/7):
				plt.scatter(node[0], node[1], color="green")
			elif float(50/7) > node[0] > float(40/7):
				plt.scatter(node[0], node[1], color="blue")
			elif float(60/7) > node[0] > float(50/7):
				plt.scatter(node[0], node[1], color="indigo")
			elif float(70/7) > node[0] > float(60/7):
				plt.scatter(node[0], node[1], color="purple")
		plt.show()


if __name__ == "__main__":
	dataset = Dataset(2)
	dataset.init_set(500)
	dataset.show()