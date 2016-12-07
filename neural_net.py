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

		conv1_output_chanel = 7	
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
		conv1_active = relu_layer.relu("conv1_active", 0.1)
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

		conv2_output_chanel = 12
		conv2_input_chanel = 7	
		conv2_kernel_shape = [2, 2]
		conv2_pad_shape = [0, 0]
		conv2_stride_shape = [1, 1]
		conv2 = conv_layer.conv2d("conv2", \
				conv2_output_chanel, \
				conv2_input_chanel, \
				conv2_kernel_shape, \
				conv2_pad_shape, \
				conv2_stride_shape)
		conv2_active = relu_layer.relu("conv2_active", 0.1)
		self.layers.append(conv2)
		self.layers.append(conv2_active)

		hidden3_num = 1000
		hidden3 = full_con_layer.full1d("hidden3", 12 * 12 * 12, hidden3_num)
		hidden3_active = sigmoid_layer.sigmoid("hidden3_active")
		self.layers.append(hidden3)
		self.layers.append(hidden3_active)
		
		output_num = 10
		output = full_con_layer.full1d("output", hidden3_num, output_num)
		output_active = sigmoid_layer.sigmoid("output_active")
		self.layers.append(output)
		self.layers.append(output_active)

		self.layer_num = len(self.layers)

	def forward(self, input):
		input_ = input
		for i in range(self.layer_num):
			self.layers[i].forward(input_)
			#print self.layers[i].name, self.layers[i].forward_output
			input_ = self.layers[i].forward_output

	def cal_output_diff(self, label):
		return self.layers[-1].forward_output - label

	def backward(self, label):
		diff_  =  self.cal_output_diff(label)
		for i in range(self.layer_num):
			self.layers[self.layer_num - i - 1].backward(diff_)
			diff_ = self.layers[self.layer_num - i - 1].backward_output

	def update(self, step):
		for i in range(self.layer_num):
			self.layers[i].update(step)
	
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
