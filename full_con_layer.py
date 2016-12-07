#coding=utf-8

import numpy as np
from util import log

class full1d:
	def __init__(self, \
			name, \
			input_dim, \
			output_dim):
		self.name = name
		self.forward_input = None
		self.forward_output = None
		self.backward_input = None
		self.backward_output = None
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.weight = self.init_weight(input_dim, output_dim)
		self.bias = self.init_bias(output_dim)
		self.need_reshape = False
		self.input_shape = None
		self.need_update = True
	
	def init_weight(self, input_dim, output_dim):
		return (np.random.rand(input_dim, output_dim) - 0.5) / 10.0	

	def init_bias(self, output_dim):
		return np.zeros((1, output_dim))

	def forward(self, input):
		if len(input.shape) == 4:
			batch_sz, input_ch, input_h, input_w = input.shape
			if self.input_dim != input_ch * input_h * input_w:
				log("%s: wrong input num [%d != %d]" % (self.name, self.input_dim, input_ch * input_h * input_w))
			self.need_reshape = True
			self.input_shape = input.shape
			self.forward_input = input.reshape(batch_sz, self.input_dim)
		else:
			self.forward_input = input

		self.forward_output = np.dot(self.forward_input, self.weight) + self.bias

	def backward(self, diff):
		self.backward_input = diff
		self.backward_output = np.dot(diff, np.transpose(self.weight))

		if self.need_reshape:
			self.backward_output = self.backward_output.reshape(self.input_shape)

	def update(self, step):
		deta_weight_sum = np.zeros((self.input_dim, self.output_dim))
		batch_size = self.forward_input.shape[0]
		for i in range(batch_size):
			deta_weight_sum += np.dot(np.transpose(self.forward_input[i:i+1, :]), self.backward_input[i:i+1, :])
		self.weight -= (step / batch_size) * deta_weight_sum
		self.bias -= (step / batch_size) * (np.cumsum(self.backward_input)[-1])
