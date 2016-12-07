#coding=utf-8

from layer import layer
import numpy as np

class conv2d(layer):
	def __init__(self, \
			name, \
			output_chanel, \
			input_chanel, \
			kernel_shape, \
			pad_shape, \
			stride_shape=[1, 1]):
		self.forward_input = None
		self.forward_output = None
		self.backward_input = None
		self.backward_output = None
		self.name = name
		self.output_ch = output_chanel
		self.input_ch = input_chanel
		self.kernel = self.init_weight(kernel_shape)
		self.bias = self.init_bias(output_chanel)
		self.kernel_shape = kernel_shape
		self.pad_shape = pad_shape
		self.stride_shape = stride_shape
	
	def init_weight(self, kernel_shape):
		return (np.random.rand(self.output_ch, \
				self.input_ch, \
				kernel_shape[0], \
				kernel_shape[1]) - 0.5) / 10.0	

	def init_bias(self, output_chanel):
		return np.zeros(output_chanel)

	def forward(self, input):
		self.forward_input = input

		batch_sz, input_ch, input_h, input_w = input.shape
		output_ch, _, kernel_h, kernel_w = self.kernel.shape
		pad_h, pad_w = self.pad_shape
		stride_h, stride_w = self.stride_shape

		input_ext_h = input_h + 2 * pad_h
		input_ext_w = input_w + 2 * pad_w
		input_ext = np.zeros((batch_sz, input_ch, input_ext_h, input_ext_w))
		for b in range(batch_sz):
			for c in range(input_ch):
				for h in range(input_h): 
					for w in range(input_w): 
						input_ext[b, c, h + pad_h, w + pad_w] = input[b, c, h, w]

		output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1
		output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1

		self.forward_output = np.zeros((batch_sz, output_ch, output_h, output_w))

		for b in range(batch_sz):
			for o_c in range(output_ch):
				o_bias = self.bias[o_c]
				for o_h in range(output_h):
					for o_w in range(output_w):
						sum_ = 0.0
						for i_c in range(input_ch):
							cur_kernel = self.kernel[o_c, i_c]
							start_h = o_h * stride_h
							start_w = o_w * stride_w
							end_h = start_h + kernel_h
							end_w = start_w + kernel_w
							sum_ +=  np.sum(input_ext[b, i_c, start_h : end_h, start_w : end_w] * cur_kernel)
						self.forward_output[b, o_c, o_h, o_w] = sum_ + o_bias

	def backward(self, diff):
		self.backward_input = diff

		batch_sz, diff_ch, diff_h, diff_w = diff.shape
		_, input_ch, kernel_h, kernel_w = self.kernel.shape
		pad_h, pad_w = self.pad_shape
		pad_h += kernel_h - 1
		pad_w += kernel_w - 1
		stride_h, stride_w = self.stride_shape

		diff_ext_h = diff_h + 2 * pad_h
		diff_ext_w = diff_w + 2 * pad_w
		diff_ext = np.zeros((batch_sz, diff_ch, diff_ext_h, diff_ext_w))
		for b in range(batch_sz):
			for c in range(diff_ch):
				for h in range(diff_h): 
					for w in range(diff_w): 
						diff_ext[b, c, h + pad_h, w + pad_w] = diff[b, c, h, w]

		output_h = (diff_h + 2 * pad_h - kernel_h) / stride_h + 1
		output_w = (diff_w + 2 * pad_w - kernel_w) / stride_w + 1
		#output_h = (diff_h - 1) * stride_h + kernel_h - 2 * pad_h
		#output_w = (diff_w - 1) * stride_w + kernel_w - 2 * pad_w

		self.backward_output = np.zeros((batch_sz, input_ch, output_h, output_w))

		# conv
		for b in range(batch_sz):
			for o_c in range(input_ch):
				for o_h in range(output_h):
					for o_w in range(output_w):
						sum_ = 0.0
						for d_c in range(diff_ch):
							cur_kernel = np.rot90(self.kernel[d_c, o_c])
							start_h = o_h * stride_h
							start_w = o_w * stride_w
							end_h = start_h + kernel_h
							end_w = start_w + kernel_w
							sum_ +=  np.sum(diff_ext[b, d_c, start_h : end_h, start_w : end_w] * cur_kernel)
						self.backward_output[b, o_c, o_h, o_w] = sum_

	def update(self, step):
		batch_sz = self.forward_input.shape[0]
		delta_kernel = np.zeros_like(self.kernel)
		for c in range(batch_sz):
			for i in range(self.output_ch):
				for j in range(self.input_ch):
					conv_ = self.forward_input[c, j]
					kernel_ = self.backward_input[c, i]
					for k_h in range(self.kernel_shape[0]):
						for k_w in range(self.kernel_shape[1]):
							start_h = k_h
							start_w = k_w
							end_h = start_h + kernel_.shape[0]
							end_w = start_w + kernel_.shape[1]
							delta_kernel[i, j, k_h, k_w] +=  np.sum(conv_[start_h : end_h, start_w : end_w] * kernel_)
		self.kernel -= step * delta_kernel / batch_sz

		delta_bias = np.zeros_like(self.bias)
		for c in range(batch_sz):
			for i in range(self.output_ch):
				delta_bias += np.sum(self.backward_input[i])
		self.bias -= step * delta_bias / batch_sz
