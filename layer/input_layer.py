#coding=utf-8

from base_layer import layer

class input2d(layer):
	def __init__(self, name):
		self.name = name
		self.forword_output = None
		self.backward_output = None	
		self.need_update = False
	
	# forward
	def forward(self, input):
		self.forward_output = input
	
	# backward
	def backward(self, diff):
		self.backward_output =  diff
	
	# update
	def update(self, step):
		pass
