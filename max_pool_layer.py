#coding=utf-8

from layer import layer
import numpy as np

class max2d(layer):
	def __init__(self, \
			name, \
			kernel_shape, \
			pad_shape, \
			stride_shape=[1, 1]):
		self.forward_input = None
		self.forward_output = None
		self.backward_input = None
		self.backward_output = None
		self.name = name
		self.kernel_shape = kernel_shape
		self.pad_shape = pad_shape
		self.stride_shape = stride_shape
		self.need_update = False

	def forward(self, input):
		self.forward_input = input

		batch_sz, input_ch, input_h, input_w = input.shape
		kernel_h, kernel_w = self.kernel_shape
		pad_h, pad_w = self.pad_shape
		stride_h, stride_w = self.stride_shape

		output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1
		output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1

		self.forward_output = np.zeros((batch_sz, input_ch, output_h, output_w))

		for	b in range(batch_sz): 
			for c in range(input_ch):
				for h in range(output_h): 
					for w in range(output_w): 
						start_h = h * stride_h - pad_h 
						start_w = w * stride_w - pad_w
						end_h = min(start_h + kernel_h, input_h) 
						end_w = min(start_w + kernel_w, input_w) 
						start_h = max(start_h, 0)
						start_w = max(start_w, 0)
						self.forward_output[b, c, h, w] = np.max(input[b, c, start_h:end_h, start_w:end_w])

	def backward(self, diff): 
		self.backward_input = diff
		
		batch_sz, diff_ch, diff_h, diff_w = diff.shape
		kernel_h, kernel_w = self.kernel_shape
		pad_h, pad_w = self.pad_shape
		stride_h, stride_w = self.stride_shape

		output_h = (diff_h - 1) * stride_h + kernel_h - 2 * pad_h
		output_w = (diff_w - 1) * stride_w + kernel_w - 2 * pad_w

		self.backward_output = np.zeros((batch_sz, diff_ch, output_h, output_w))
		
		for b in range(batch_sz):
			for c in range(diff_ch):
				for h in range(diff_h): 
					for w in range(diff_w): 
						start_h = h * stride_h
						start_w = w * stride_w
						end_h = min(start_h + kernel_h, output_h)
						end_w = min(start_w + kernel_w, output_w)
						for h_ in range(start_h, end_h):
							for w_ in range(start_w, end_w):
								self.backward_output[b, c, h_, w_] += \
										int(self.forward_input[b, c, h_, w_] == self.forward_output[b, c, h, w]) \
										* diff[b, c, h, w]
	def update(self, step):
		pass
