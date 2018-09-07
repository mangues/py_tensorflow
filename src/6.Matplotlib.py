# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# 创建数据
x = np.linspace(-2, 2, 100)  #-2到2之间创建100个数据
y1 = 3 * x + 1;
y2 = x**5 + 1;

# 创建图像
plt.plot(x,y1,color="red",linewidth=5.0,linestyle="--")
plt.plot(x,y2)

#显示图像
plt.show()



