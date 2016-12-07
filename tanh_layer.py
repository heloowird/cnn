#coding=utf-8

from layer import layer
import numpy as np

class tanh(layer):
	def __init__(self, name):
		self.name = name
		self.forword_output = None
		self.backward_output = None

	# forward
	def forward(self, input):
		self.forward_output = ((math.exp(input) - math.exp(-1.0 * input)) / (math.exp(input) + math.exp(-1.0 * input)))
	
	# backward
	def backward(self, diff):
		self.backward_output =  ((1.0 - math.pow(self.forward_output, 2)) * diff)
	
	def update(self, step):
		pass
