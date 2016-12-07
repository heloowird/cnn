#coding=utf-8

import numpy as np

import input_layer
import conv_layer
import max_pool_layer
import mean_pool_layer
import full_con_layer
import sigmoid_layer
import tanh_layer
import relu_layer

import util

class cnn:
	def __init__(self):
		self.layers = []

		input = input_layer.input2d("input")
		self.layers.append(input)

		conv1_output_chanel = 2	
		conv1_input_chanel = 1	
		conv1_kernel_shape = [3, 3]
		conv1_pad_shape = [0, 0]
		conv1_stride_shape = [1, 1]
		conv1 = conv_layer.conv2d("conv1", \
				conv1_output_chanel, \
				conv1_input_chanel, \
				conv1_kernel_shape, \
				conv1_pad_shape, \
				conv1_stride_shape)
		conv1_active = relu_layer.relu("conv1_active")
		self.layers.append(conv1)
		self.layers.append(conv1_active)

		pool1_kernel_shape = [2, 2]
		pool1_pad_shape = [0, 0]
		pool1_stride_shape = [2, 2]
		pool1 = max_pool_layer.max2d("pool1", \
				pool1_kernel_shape, \
				pool1_pad_shape, \
				pool1_stride_shape)
		self.layers.append(pool1)

		output_num = 4
		output = full_con_layer.full1d("output", 8, output_num)
		output_active = sigmoid_layer.sigmoid("output_active")
		self.layers.append(output)
		self.layers.append(output_active)

		self.layer_num = len(self.layers)

	def forward(self, input):
		input_ = input
		for i in range(self.layer_num):
			self.layers[i].forward(input_)
			print self.layers[i].name, "forward_output", self.layers[i].forward_output
			input_ = self.layers[i].forward_output

	def cal_output_diff(self, label):
		return self.layers[-1].forward_output - label

	def backward(self, label):
		diff_  =  self.cal_output_diff(label)
		print "final_diff", diff_
		for i in range(self.layer_num):
			self.layers[self.layer_num - i - 1].backward(diff_)
			print self.layers[self.layer_num -i - 1].name, "backward_output", self.layers[self.layer_num - i -1].backward_output
			diff_ = self.layers[self.layer_num - i - 1].backward_output

	def update(self, step):
		for i in range(self.layer_num):
			if self.layers[i].need_update:
				print self.layers[i].name, "update_before weight", self.layers[i].weight
				print self.layers[i].name, "update_before bias", self.layers[i].bias
				self.layers[i].update(step)
				print self.layers[i].name, "update_after weight", self.layers[i].weight
				print self.layers[i].name, "update_after bias", self.layers[i].bias
	
	def train(self, data, label, step):
		self.forward(data)
		self.backward(label)
		self.update(step)	

	def predict(self, data):
		self.forward(data)
		predict_label = np.argmax(self.layers[-1].forward_output, axis=1)
		predict_label = predict_label.astype(int)
		return predict_label

	def accuracy(self, data, label):
		self.forward(data)
		predict_label = np.argmax(self.layers[-1].forward_output, axis=1)
		true_label = np.argmax(label, axis=1)
		right_cnt = np.sum((predict_label - true_label) == 0)
		return 1.0 * right_cnt / data.shape[0]
