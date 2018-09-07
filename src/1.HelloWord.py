# -*- coding: utf-8 -*-
# Python
import tensorflow as tf
hello = tf.constant('你好, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
print('你好')

