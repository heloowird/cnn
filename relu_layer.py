#coding=utf-8

from layer import layer
import numpy as np

class relu(layer):
	def __init__(self, name, negative_slope=0.0):
		self.name = name
		self.forword_output = None
		self.backward_output = None
		self.negative_slope = negative_slope

	# forward
	def forward(self, input):
		self.forward_output = (input >= 0.0).astype(int) * input \
				+  (input < 0.0).astype(int) * input * self.negative_slope
	
	# backward
	def backward(self, diff):
		self.backward_output =  ((self.forward_output >= 0.0).astype(int) \
				+ (self.forward_output < 0.0).astype(int) * self.negative_slope) * diff

	# update
	def update(self, step):
		pass
