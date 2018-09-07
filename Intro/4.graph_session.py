# -*- coding: utf-8 -*-
import tensorflow as tf

constant1 = tf.constant([[2, 2]])
constant2 = tf.constant([[4], [4]])

#矩阵乘法 matrix multiple
multiple = tf.matmul(constant1, constant2)

session = tf.Session()
print(session.run(multiple))


session.close()

if constant1.graph is tf.get_default_graph():
    print("constant1所在图是默认上下文图")


# 管理上下文，自动close
with tf.Session() as se:
    print(multiple)
    print(se.run(multiple))

tf.summary.FileWriter("/Users/mangues/Desktop/dep/py-tensorflow/logs",session.graph);
