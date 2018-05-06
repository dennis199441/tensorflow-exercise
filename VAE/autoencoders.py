import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


# Loss function
log_likelihood = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
KL_divergence = -0.5 * tf.reduce_sum(1 + 2 * logstd - tf.pow(mean, 2) - tf.exp(2 * logstd), reduction_indices=1)

variational_lower_bound = tf.reduce_mean(log_likelihood - KL_divergence)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)


# Train
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver()

import time #lets clock training time..

num_iterations = 1000000
recording_interval = 1000
#store value for these 3 terms so we can plot them later
variational_lower_bound_array = []
log_likelihood_array = []
KL_divergence_array = []
iteration_array = [i * recording_interval for i in range(int(num_iterations / recording_interval))]
for i in range(num_iterations):
    # np.round to make MNIST binary
    #get first batch (200 digits)
    x_batch = np.round(mnist.train.next_batch(200)[0])
    #run our optimizer on our data
    sess.run(optimizer, feed_dict={X: x_batch})
    if (i % recording_interval == 0):
        #every 1K iterations record these values
        vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
        save_path = saver.save(sess, "models/autoencoder.ckpt")
        print("saved to %s" % save_path)
        print ("Iteration: {}, Loss: {}".format(i, vlb_eval))
        variational_lower_bound_array.append(vlb_eval)
        log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
        KL_divergence_array.append(np.mean(KL_divergence.eval(feed_dict={X: x_batch})))

save_path = saver.save(sess, "models/autoencoder.ckpt")
print("saved to %s" % save_path)


plt.figure()
#for the number of iterations we had 
#plot these 3 terms
plt.plot(iteration_array, variational_lower_bound_array)
plt.plot(iteration_array, KL_divergence_array)
plt.plot(iteration_array, log_likelihood_array)
plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
plt.title('Loss per iteration')
plt.show()