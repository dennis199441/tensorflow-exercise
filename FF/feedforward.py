import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def weight_variable(shape, name):
	initial = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def fully_connected_layer(X, weight, bias):
	return tf.matmul(X, weight) + bias

def get_iris_data():
	iris = datasets.load_iris()
	data = iris['data']
	target = iris['target']

	num_labels = len(np.unique(target))
	target = np.eye(num_labels)[target]

	return train_test_split(data, target, test_size=0.3, random_state=RANDOM_SEED)

def train():
	train_X, test_X, train_y, test_y = get_iris_data()

	input_layer_size = train_X.shape[1]
	hidden_layer_size = 256
	output_layer_size = train_y.shape[1]

	X = tf.placeholder(tf.float32, shape=([None, input_layer_size]))
	y = tf.placeholder(tf.float32, shape=([None, output_layer_size]))

	# input layer
	weight_1 = weight_variable(shape=[input_layer_size, hidden_layer_size], name='weight_1')
	bias_1 = bias_variable(shape=[1, hidden_layer_size], name='bias_1')
	hidden = tf.nn.sigmoid(fully_connected_layer(X, weight_1, bias_1))

	# hidden layer
	weight_2 = weight_variable(shape=(hidden_layer_size, output_layer_size), name='weight_2')
	bias_2 = bias_variable(shape=(1, output_layer_size), name='bias_2')

	y_hat = tf.matmul(hidden, weight_2) + bias_2
	predict = tf.argmax(y_hat, axis=1)

	# cost function
	cost = tf.reduce_min(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat))
	update = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	# training loop
	_EPOCH = 1000
	train_accuracy = 0
	test_accuracy = 0
	for epoch in range(_EPOCH):
		for i in range(len(train_X)):
			sess.run(update, feed_dict={X: train_X[i:i+1], y: train_y[i:i+1]})

		train_accuracy += np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
		test_accuracy += np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
		
	print("Number of epoch = %d, avg. train accuracy = %.2f%%, avg. test accuracy = %.2f%%" % (_EPOCH, 100. * train_accuracy/_EPOCH, 100. * test_accuracy/_EPOCH))
	test = sess.run(predict, feed_dict={X: [test_X[0]], y: [test_y[0]]})
	print('testX: ', [test_X[0]])
	print('testy: ', [test_y[0]])
	print('test: ', test)

	sess.close()

if __name__ == '__main__':
	train()



