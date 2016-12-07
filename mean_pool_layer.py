#coding=utf-8

from layer import layer
import numpy as np

class mean2d(layer):
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

	def forward(self, input):
		self.forward_input = input

		batch_sz, input_ch, input_h, input_w = input.shape
		kernel_h, kernel_w = self.kernel_shape
		pad_h, pad_w = self.pad_shape
		stride_h, stride_w = self.stride_shape

		output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1
		output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1
		while output_h and (output_h -1) * stride_h >= input_h + pad_h:
			output_h -= 1
		while output_w and (output_w -1) * stride_w >= input_w + pad_w:
			output_w -= 1

		self.forward_output = np.zeros((batch_sz, input_ch, output_h, output_w))

		for	b in range(batch_sz): 
			for c in range(input_ch):
				for h in range(output_h): 
					for w in range(out_width): 
						start_h = h * stride_h - pad_h 
						start_w = w * stride_w - pad_w
						end_h = min(start_h + kernel_h, input_h + pad_h) 
						end_w = min(start_w + kernel_w, input_w + pad_w) 
						pool_size = (end_h - start_h) * (end_w - start_w)
						start_h = max(start_h, 0)
						start_w = max(start_w, 0)
						hend_h = min(end_h, input_h)
						hend_w = min(end_w, input_w)
						self.forward_output[b, c, h, w] = np.sum(input[b, c, start_h:end_h, start_w:end_h]) / pool_size

	def backward(self, diff): 
		self.backward_input = diff

		batch_sz, diff_ch, diff_h, diff_w = diff.shape
		kernel_h, kernel_w = self.kernel_shape
		pad_h, pad_w = self.pad_shape
		pad_h += kernel_h - 1
		pad_w += kernel_w - 1
		stride_h, stride_w = self.stride_shape
		pool_size = kernel_h * kernel_w

		output_h = (diff_h - 1) * stride_h + kernel_h - 2 * pad_h
		output_w = (diff_w - 1) * stride_w + kernel_w - 2 * pad_w

		self.forward_output = np.zeros((batch_sz, diff_ch, output_h, output_w))
		
		for b in range(batch_size):
			for c in range(diff_ch):
				for h in range(diff_h): 
					for w in range(diff_w): 
						start_h = h * stride_h
						start_w = w * stride_w
						end_h = min(start_h + kernel_h, output_h)
						end_w = min(start_w + kernel_w, output_w)
						for h_ in range(start_h, end_h):
							for w_ in rangw(start_w, end_w):
								self.backward_output[b, c, h_, w_] +=  diff[b, c, h, w] / pool_size
	
	def update(self, step):
		pass
