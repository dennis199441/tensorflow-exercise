import tensorflow as tf
from trainGAN import *

# sess = tf.Session()
# z_dimensions = 100
# z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])
batch_size = 16
tf.reset_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

z_dimensions = 100
x_placeholder = tf.placeholder("float", shape = [None,28,28,1]) #Placeholder for input images to the discriminator
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions]) #Placeholder for input noise vectors to the generator

Dx = discriminator(x_placeholder) #Dx will hold discriminator outputs (unnormalized) for the real MNIST images
Gz = generator(z_placeholder, batch_size, z_dimensions) #Gz holds the generated images
# Dg = discriminator(Gz, reuse=True) #Dg will hold discriminator outputs (unnormalized) for generated images

# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

# d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
# d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
# d_loss = d_loss_real + d_loss_fake

# tvars = tf.trainable_variables()
# d_vars = [var for var in tvars if 'd_' in var.name]
# g_vars = [var for var in tvars if 'g_' in var.name]


saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./models/gans.ckpt.meta')
saver.restore(sess, './models/gans.ckpt')

sample_image = generator(z_placeholder, 1, z_dimensions, reuse=True)
z_batch = np.random.normal(-1, 1, [1,z_dimensions])
# z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
# sess.run(tf.global_variables_initializer())
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')
plt.show()
