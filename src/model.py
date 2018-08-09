import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import dataset


class Model:
	def __init__(self, dataset, dimension, C):
		self.dataset = dataset
		self.dimension = dimension
		self.a = np.array([0 for _ in range(dataset.set.__len__())], dtype=float)
		self.w = np.array([0 for _ in range(dimension)], dtype=float)
		self.b = 0
		self.C = C
	
	def kkt(self, i):
		# 违背KKT条件越大，返回值越大
		false = 0
		if self.dataset.set[i].label * self.dataset.set[i].predict <= 1 and self.a[i] < self.C:
			false += 1
		if self.dataset.set[i].label * self.dataset.set[i].predict >= 1 and self.a[i] > 0:
			false += 1
		if self.dataset.set[i].label * self.dataset.set[i].predict == 1 and (self.a[i] == 0 or self.a[i] == self.C):
			false += 1
		return false
		
	def function(self, i):
		xi = self.dataset.set[i].data
		return np.dot(self.w.T, xi)+self.b
	
	def init(self):
		for i in range(len(self.dataset.set)):
			self.dataset.set[i].predict = np.dot(self.a.T, self.dataset.set[i].data)
			self.dataset.set[i].refresh_e()
		
	def smo(self):
		# 第一轮
		out = 0
		for i in range(len(self.dataset.set)):
			self.dataset.set[i].predict = self.function(i)
			self.dataset.set[i].refresh_e()
		for i in range(len(self.dataset.set)):
			node_i = i
			node_j = self.select_a2(node_i)
			self.update(node_i, node_j)
		# 第二轮
		round_num = 0
		while True:
			round_num += 1
			temp_a = copy.copy(self.a)
			node_i = self.select_a1()
			node_j = self.select_a2(node_i)
			self.update(node_i, node_j)
			if (self.a == temp_a).all():
				out += 1
			else:
				out = 0
			if out == 3:
				break
			if round_num >= 1000000:
				break
		
	@staticmethod
	def kernel_function(xi, xj, k="liner"):
		if k == "liner":
			return np.dot(xi.T, xj)
	
	def select_a1(self):
		temp_false = 0
		temp_node = random.randint(0, len(self.dataset.set))
		for i in range(len(self.dataset.set)):
			if self.kkt(i) > temp_false and 0 < self.a[i] < self.C:
				temp_node = i
				temp_false = self.kkt(i)
			if temp_false == 3:
				return temp_node
		return temp_node
	
	def select_a2(self, j):
		max_num = 0
		temp_node = j
		for i in range(len(self.dataset.set)):
			if abs(self.dataset.set[i].e - self.dataset.set[j].e) >= max_num:
				temp_node = i
				max_num = abs(self.dataset.set[i].e - self.dataset.set[j].e)
		return temp_node
	
	def update(self, i, j, kernel_flag="liner"):
		ai_old = self.a[i]
		aj_old = self.a[j]
		
		# 求解a2_new的上下界
		if self.dataset.set[i].label != self.dataset.set[j].label:
			L = max(0, aj_old - ai_old)
			H = min(self.C, self.C + aj_old - ai_old)
		else:
			L = max(0, ai_old + aj_old - self.C)
			H = min(self.C, ai_old + aj_old)
		# 求出不准确的a2_new
		yita = self.kernel_function(self.dataset.set[i].data, self.dataset.set[i].data, kernel_flag) + self.kernel_function(self.dataset.set[j].data, self.dataset.set[j].data, kernel_flag) - 2*self.kernel_function(self.dataset.set[i].data, self.dataset.set[j].data, kernel_flag)
		aj_new_temp = aj_old + self.dataset.set[j].label * (self.dataset.set[i].e - self.dataset.set[j].e) / yita
		# 求出准确的a2_new
		if aj_new_temp > H:
			aj_new = H
		elif L <= aj_new_temp <= H:
			aj_new = aj_new_temp
		else:
			aj_new = L
		# 求解准确的a1_new
		ai_new = ai_old + self.dataset.set[i].label * self.dataset.set[j].label * (aj_old - aj_new)
		# 更新 b, ai, aj
		b_old = self.b
		b1_new = b_old - self.dataset.set[i].e - self.dataset.set[i].label * (ai_new - ai_old) * self.kernel_function(self.dataset.set[i].data, self.dataset.set[i].data, kernel_flag) - self.dataset.set[j].label * (aj_new - aj_old) * self.kernel_function(self.dataset.set[j].data, self.dataset.set[j].data, kernel_flag)
		b2_new = b_old - self.dataset.set[j].e - self.dataset.set[i].label * (ai_new - ai_old) * self.kernel_function(self.dataset.set[i].data, self.dataset.set[i].data, kernel_flag) - self.dataset.set[j].label * (aj_new - aj_old) * self.kernel_function(self.dataset.set[j].data, self.dataset.set[j].data, kernel_flag)
		if 0 < ai_new < self.C:
			self.b = b1_new
		elif 0 < aj_new < self.C:
			self.b = b2_new
		else:
			self.b = (b1_new + b2_new) / 2
		self.a[i] = ai_new
		self.a[j] = aj_new
		self.update_w()
		for i in range(len(self.dataset.set)):
			self.dataset.set[i].predict = self.function(i)
			self.dataset.set[i].refresh_e()
	
	def update_w(self):
		self.w = self.w = np.array([0 for _ in range(self.dimension)], dtype=float)
		for i in range(len(self.dataset.set)):
			self.w += self.a[i] * self.dataset.set[i].label * self.dataset.set[i].data
	
	def show(self):
		for i in range(len(self.dataset.set)):
			if 0 < self.a[i] < self.C:
				plt.scatter(self.dataset.set[i].data[0], self.dataset.set[i].data[1], color="red")
			else:
				plt.scatter(self.dataset.set[i].data[0], self.dataset.set[i].data[1], color="blue")
		plt.show()
			

if __name__ == "__main__":
	data_set = dataset.Dataset(2)
	data_set.init_set(500)
	model = Model(data_set, 2, 1)
	model.smo()
	model.show()
