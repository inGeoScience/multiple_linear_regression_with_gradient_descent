import numpy
import pandas
from matplotlib import pyplot

# 数据读取
data = pandas.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
# Mean Normalization
data = (data - data.mean()) / data.std()
# 构造特征值矩阵、实际值矩和参数的二维数组
data.insert(0, 'ones', 1)
colNum = data.shape[1]
X = data.iloc[:, 0:colNum-1] # 这一步得到的是DataFrame
X = X.to_numpy()
Y = data.iloc[:, colNum-1:colNum]
Y = Y.to_numpy()
theta = numpy.zeros((1, colNum-1)) # theta为(1, 3)的二维数组。共有3个参数，包括截距项b在内
# 代价函数
def compCost(X, Y, theta):
    inner = numpy.power(numpy.matmul(X, theta.T) - Y, 2)
    return 1/(2*len(X))*numpy.sum(inner)
# Batch Gradient Descent Function
def batchGradientDescent(X, Y, theta, alpha, iters):
    # 创建临时参数二维数组
    temp_theta = numpy.zeros(theta.shape)
    # 创建代价列表(一维数组)以记录每一次代价，这里当作list来用
    cost = numpy.zeros(iters)
    for i in range(iters):
        error = numpy.matmul(X, theta.T) - Y
        # 计算每一个θ，并把θ添加到临时参数向量里。enumerate()返回一个tuple，包含两个元素——第i行和值。
        for nth_para, para in enumerate(theta.T):
            # 每次累加中乘的x^(i)是一个标量，所以用的numpy.multiply()，而不是*
            # X[:, nth_para]是一个一维数组，为了得到对应元素的点乘，把它转换成二维数组。
            # 在main.py中，X是矩阵。X[:, nth_para]也是一个矩阵
            para = para - ((alpha/len(X))*numpy.sum(numpy.multiply(error, X[:, nth_para].reshape(len(X), 1))))
            temp_theta[0, nth_para] = para
        # 同步更新，并记录每一次代价
        theta = temp_theta
        cost[i] = compCost(X, Y, theta)
    return theta, cost
# 给定学习率、迭代次数
alpha = 0.01
iters = 1000
# 进行梯度下降
theta, cost = batchGradientDescent(X, Y, theta, alpha, iters)
print(theta)
print(cost[-1])
# 出图
fig, ax = pyplot.subplots()
ax.plot(numpy.arange(iters), cost, 'r')
ax.set_title('Multiple Linear Regression with Gradient Descent')
ax.set_xlabel('epochs')
ax.set_ylabel('cost')
pyplot.show()