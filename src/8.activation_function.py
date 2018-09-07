# -*- coding: utf-8 -*-
# 激活函数
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#创建输入数据
x = np.linspace(-7,7,180) #-7,7 180个点

def sigmoid(inputs):
    y = [1/float(1+np.exp(-x)) for x in inputs]
    return y

def relu(inputs):
    y = [x * (x>0) for  x in inputs]
    return y

def tanh(inputs):
    y = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x)) for x in inputs]
    return y

def softplus(inputs):
    y = [np.log(1+np.exp(x)) for x in inputs]
    return y


def show(location,y,minY,MaxY,label):
    plt.subplot(location)
    plt.plot(x, y, c='red', label=label)
    plt.ylim(minY, MaxY)
    plt.legend(loc="best")


# 经过 TensorFlow 的激活函数处理的各个Y值
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

#创建会话
with tf.Session() as sess:
    y_sigmoid,y_relu,y_tanh,y_softplus = sess.run([y_sigmoid,y_relu,y_tanh,y_softplus])
    print(y_sigmoid,y_relu,y_tanh,y_softplus)
    #创建激活函数的图像
    show(221,y_sigmoid,-0.2,1.2,"sigmoid");
    show(222,y_relu,-1,6,"relu");
    show(223,y_tanh,-1.3,1.3,"tanh");
    show(224,y_softplus,-1,6,"softplus");
    plt.show()
