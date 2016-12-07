#coding=utf-8

import os
import time

import numpy as np
import pandas as pd

import neural_net

np.set_printoptions(threshold='nan') 

def load_pre_train_weight(filename, shape):
	if os.path.isfile(filename):
		weight = np.fromfile(filename, dtype=np.float)
		weight.shape = shape
		return weight
	else:
		return None

def load_one_hot_data(filename):
	raw_train_data = pd.read_csv(filename)
	train_data = raw_train_data.values[:]
	return train_data

def labels_to_one_hot(raw_labels):
	labels = np.zeros((raw_labels.shape[0],  10))
	labels[np.arange(raw_labels.shape[0]), raw_labels] = 1
	labels = labels.astype(np.float32)
	return labels

def gen_batch_data(train_data, batch_size):
	batch_index = np.random.randint(0, train_data.shape[0], batch_size)
	batch_train_data = train_data[batch_index]
	return ((batch_train_data[:, 1:] / 255.0).reshape((batch_size, 1, 28, 28)), labels_to_one_hot(batch_train_data[:, 0]))

def train_cnn(step_num, step):
	raw_train_data = load_one_hot_data("./data/train.csv")

	feature_num = raw_train_data.shape[1] - 1
	label_num = 10
	batch_size = 20

	print "feature_num: %d" % feature_num
	print "label_num: %d" % label_num
	print "batch_size: %d" % batch_size

	# initialize simple cnn
	cnn_net = neural_net.cnn()

	# train
	np.random.shuffle(raw_train_data)
	train_len = int(raw_train_data.shape[0] * 0.95)
	train_data, valid_data = raw_train_data[:train_len], raw_train_data[train_len:]
	valid_features, valid_labels = valid_data[:, 1:] / 255.0, labels_to_one_hot(valid_data[:, 0])
	
	display_step = 1
	for i in xrange(step_num):
		features_data, labels_data = gen_batch_data(train_data, batch_size)
		start_t = time.time()
		cnn_net.train(features_data, labels_data, step)
		print "%d batch cost time:  %d ms" % (i, int((time.time() - start_t)*1000))

		# print accuracy of every one epoch
		if i % display_step == 0 or (i+1) == step_num:
			print "accuracy of %d step (%d epoch): %f" % ((i+1), (i+1)/840, cnn_net.accuracy(valid_features.reshape((-1, 1, 28, 28)), valid_labels))

		if i % (display_step * 10) == 0 and i:
			display_step *= 10

	# test
	#test_data = load_one_hot_data("./data/test.csv")
	#predict_label = cnn_net.predict((test_data / 255.0).reshape((-1, 1, 28, 28)))
	#predict_label = [np.arange(1, 1+len(predict_label)), predict_label]
	#predict_label = np.transpose(predict_label)
	#np.savetxt('submission.csv', predict_label, fmt='%i,%i', header='ImageId,Label', comments='')

if __name__ == "__main__":
	train_cnn(1000, 1e-4)
