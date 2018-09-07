# -*- coding: utf-8 -*-
# 用梯度下降的优化方法来快速解决线性回归问题
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#构建数据
points_num = 100
vectors = []
# 用Numpy的正态随机分布函数生成100个点，这些点（x,y）坐标值对应线性方程 y = 0.1 * x + 0.2
# 权重（Weight）0.1，偏差(Bias)0.2

for i in range(points_num): #返回的是一个对象，并没有将数据完全实例化，所以内存中只有一个对象的空间，对性能优化还是很有帮助的。
    x1 = np.random.normal(0.0,0.66)
    y1 = 0.1 * x1 + 0.2+np.random.normal(0.0,0.04)
    vectors.append([x1,y1])

x_data = [v[0] for v in vectors]   #真实的点的x坐标
y_data = [v[1] for v in vectors]  #真实的点的y坐标

# 图像1：展示所有的100个随机数据点
plt.plot(x_data,y_data,'r.',label="Original data")  #红色星型的点 原始的点
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()


#构建线性回归模型
W = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #初始化 Weight
b = tf.Variable(tf.zeros([1]))  #初始化 Bias
y = W * x_data +b

#定义 loss function(损失函数) 或者 cost function （代价函数）
#对 Tensor 的所有纬度计算((y-y_data) ^ 2) 之和 / N
loss = tf.reduce_mean(tf.square(y - y_data)) # 平方和的平均

#用梯度下降的优化器来优化我们的loss function
optimizer = tf.train.GradientDescentOptimizer(0.5)  #学习效率 梯度下降的步长
train = optimizer.minimize(loss)  # 损失最小

# 初始化变量
init  = tf.global_variables_initializer()

#创建会话
with tf.Session() as sess:
    sess.run(init)
    # 训练20步
    for step in range(20):
        #优化每一步
        sess.run(train)
        #打印出每一步的损失，权重和偏差
        print("Strp=%d,Loss=%f,[Weight=%f Bias=%f]" % (step,sess.run(loss),sess.run(W),sess.run(b)))

    #图像2 ： 绘制所有的点并且会支出最佳拟合的直线
    plt.plot(x_data,y_data,'r.',label="Original data")  #红色星型的点 原始的点
    plt.title("Linear Regression using Gradient Descent")
    plt.plot(x_data,sess.run(W)*x_data+sess.run(b),label="Fitted line") #拟合的线
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



