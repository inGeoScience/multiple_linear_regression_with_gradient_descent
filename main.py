import numpy
import pandas
from matplotlib import pyplot

# 数据读取
data = pandas.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
# Mean Normalization
data = (data - data.mean()) / data.std()
# 构造特征值的矩阵、实际值的矩阵和参数矩阵
data.insert(0, 'ones', 1)
colNum = data.shape[1]
X = data.iloc[:, 0:colNum-1] # 这一步得到的是DataFrame
X = numpy.matrix(X.values) # 将DataFrame转为Matrix
Y = data.iloc[:, colNum-1:colNum]
Y = numpy.matrix(Y.values)
theta = numpy.matrix(numpy.zeros(3)) # 有三个参数θ待求，转换成矩阵才能参与运算
# 代价函数
def compCost(X, Y, theta):
    inner = numpy.power((X * theta.T) - Y, 2)
    return 1/(2*len(X))*numpy.sum(inner)
# Batch Gradient Descent Function
def batchGradientDescent(X, Y, theta, alpha, iters):
    # 创建临时参数矩阵
    temp_theta = numpy.matrix(numpy.zeros(theta.shape))
    # 创建代价列表(ndarray)以记录每一次代价
    cost = numpy.zeros(iters)
    for i in range(iters):
        error = X*theta.T - Y
        # 计算每一个θ，并把θ添加到临时矩阵里
        for nth_para, para in enumerate(theta.T):
            # 每次累加中乘的x^(i)是一个标量，所以用的numpy.multiply()，而不是*
            para = para - ((alpha/len(X))*numpy.sum(numpy.multiply(error, X[:, nth_para])))
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