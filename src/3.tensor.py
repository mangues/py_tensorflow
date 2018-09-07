import tensorflow as tf
import numpy as np
const = tf.constant(3)
print(const)

x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
     # print(sess.run(y))  # ERROR: will fail because x was not fed.
     rand_array = np.random.rand(1024, 1024)
     print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.