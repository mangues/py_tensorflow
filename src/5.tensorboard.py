# -*- coding: utf-8 -*-
import tensorflow as tf
#方程 线性y = W * x +b
W = tf.Variable(2.0,dtype=tf.float32,name="Weight") #权重
b = tf.Variable(1.0,dtype=tf.float32,name="Bias") # 偏差
x = tf.placeholder(dtype=tf.float32,name="input") # 输入

with tf.name_scope("output"): #输出的命名空间
    y = W * x + b  #输出

# 定义日志保存路径
path = "../logs"
# 创建用于初始化所有变量（Variable）的操作
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)  #初始化W，b变量
    writer = tf.summary.FileWriter(path,sess.graph);
    result = sess.run(y,{x:3.0})   # x赋值3.0
    print("y=%s" % result)  #打印

