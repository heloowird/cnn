#coding=utf-8

import numpy as np

from base_layer import layer

class tanh(layer):
	def __init__(self, name):
		self.name = name
		self.forword_output = None
		self.backward_output = None
		self.need_update = False

	# forward
	def forward(self, input):
		self.forward_output = ((np.exp(input) - np.exp(-1.0 * input)) / (np.exp(input) + np.exp(-1.0 * input)))
	
	# backward
	def backward(self, diff):
		self.backward_output =  ((1.0 - np.power(self.forward_output, 2)) * diff)
	
	def update(self, step):
		pass
