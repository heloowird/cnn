#coding=utf-8

from layer import layer
import numpy as np

class sigmoid(layer):
	def __init__(self, name):
		self.name = name
		self.forward_output = None
		self.backward_output = None
		self.need_update = False

	# forward
	def forward(self, input):
		self.forward_output = 1.0 / (np.exp(-1.0 * input) + 1.0)
	
	# backward
	def backward(self, diff):
		self.backward_output = diff * self.forward_output * (1.0 - self.forward_output)
	
	def update(self, step):
		pass
