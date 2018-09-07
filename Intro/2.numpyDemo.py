import numpy as np
vector = np.array([1,2,3])
print(vector.shape)
print(vector.size)
print(vector.ndim)
print(type(vector))
matrix = np.array([[1,2],[3,4]])
print(matrix.shape)
print(matrix.size)
print(matrix.ndim)
print(type(matrix))


zeros = np.zeros((3,4))
print(zeros)

ones = np.ones((3,4))
print(ones)

# 对角线矩阵
eyes = np.eye(4)
print(eyes)
