import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./tmp/data', one_hot=True)

n_pixels = 28 * 28

X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

def weight_variables(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def fully_connected_layer(X, W, b):
	return tf.matmul(X, W) + b

latent_dim = 20
h_dim = 500

# encoder layer: tanh activation function
# layer 1
W_encoder = weight_variables([n_pixels, h_dim], 'W_encoder')
b_encoder = bias_variable([h_dim], 'b_encoder')
h_encoder = tf.nn.tanh(fully_connected_layer(X, W_encoder, b_encoder))

# layer 2: hidden layer
W_mean = weight_variables([h_dim, latent_dim], 'W_mean')
b_mean = bias_variable([latent_dim], 'b_mean')
mean = fully_connected_layer(h_encoder, W_mean, b_mean)

W_logstd = weight_variables([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = fully_connected_layer(h_encoder, W_logstd, b_logstd)

# Add RANDOMESS
noise = tf.random_normal([1, latent_dim])
# z is the ultimate output of our encoder
z = mean + noise * tf.exp(0.5 * logstd)

# decoder: tanh activation function
# layer 1
W_decoder = weight_variables([latent_dim, h_dim], 'W_decoder')
b_decoder = bias_variable([h_dim], 'b_decoder')
h_decoder = tf.nn.tanh(fully_connected_layer(z, W_decoder, b_decoder))

# layer 2: hidden layer
W_reconstruct = weight_variables([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')
reconstruction = tf.nn.sigmoid(fully_connected_layer(h_decoder, W_reconstruct, b_reconstruct))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./models/autoencoder.ckpt.meta')
saver.restore(sess, "./models/autoencoder.ckpt")

num_pairs = 10
image_indices = np.random.randint(0, 200, num_pairs)
#Lets plot 10 digits
for pair in range(num_pairs):
    #reshaping to show original test image
    x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
    plt.figure()
    x_image = np.reshape(x, (28,28))
    plt.subplot(121)
    plt.imshow(x_image)
    #reconstructed image, feed the test image to the decoder
    x_reconstruction = reconstruction.eval(session=sess, feed_dict={X: x})
    #reshape it to 28x28 pixels
    x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
    #plot it!
    plt.subplot(122)
    plt.imshow(x_reconstruction_image)

plt.show()