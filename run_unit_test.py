#coding=utf-8

import os
import time

import numpy as np
import pandas as pd

import neural_net_test

np.set_printoptions(threshold='nan') 

def train_cnn(step):
	# initialize simple cnn
	cnn_net = neural_net_test.cnn()

	features_data = np.arange(36).reshape(1, 1, 6, 6).astype("float")

	features_data /= 35.0
	labels_data = np.zeros((1, 4))
	labels_data[0, 2] = 1
	
	cnn_net.train(features_data, labels_data, step)

if __name__ == "__main__":
	train_cnn(1e-4)
