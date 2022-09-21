


import pandas as pd
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = np.loadtxt('iris.txt', delimiter = ",", dtype = np.str)
for i in range (len(data)):
    data[i][-1] = ('1', '0')[data[i][-1] == 'Iris-versicolor']
data = data.astype(np.float)
data_frame = pd.DataFrame(data)
data_frame.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data_frame = data_frame[['petal length', 'petal width', 'class']]
data_frame = data_frame[50 : ]

# normalize data
normalized_data = (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())

# inserting bias feature
normalized_data.insert(0, "bias", 1)

versicolor = normalized_data[ : 50]
virginica = normalized_data[50 : ]

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(versicolor['petal length'], versicolor['petal width'], label = 'Iris-versicolor')
plt.scatter(virginica['petal length'], virginica['petal width'], color = 'red', label = 'Iris-virginica')
X = normalized_data[['bias', 'petal length', 'petal width']].values
Y = normalized_data[['class']].values


X, x_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)



epochs = 5000
alpha = 0.01
theta = [1, 1, 1]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
p = (10e+4)/0.3
def computeLoss(x) :
    Loss = 0
    for i in range (len(X)) :
        y = Y[i][0]
        Loss += (y - (sigmoid(theta[2] * X[i][2] + theta[1] * X[i][1] + theta[0] * X[i][0]))) * X[i][x]
    return Loss

def gradientAscent() :
    for i in range (epochs):
        for j in range(len(theta)):
            theta[j] = theta[j] + alpha * computeLoss(j)
    return(theta)

def probability(x):
    return sigmoid(np.matmul(x, theta))
def predict(x, threshold = 0.5):
    # zeros = 0
    # ones = 0
    lst=[]
    prob = probability(x)
    for i in range(len(prob)) :
        if (prob[i] >= threshold):
            # print(1)
            lst.append(1)
            # ones += 1
        else :
            # print(0)
            lst.append(0)
            # zeros += 1
    # print("zeros: %d & ones: %d" % (zeros, ones))
    return lst

#Calculate Error
def MSE(thetaF, x, y):
    tError = 0
    yhat = np.matmul(x , thetaF)
    yy = yhat-y
    ys = yy**2
    ys2 = np.sum(ys)/(len(x)*p)
    tError = ys2

    return tError




gradientAscent()
predict(X)


errtrain = MSE(theta, X, Y)
print("train MSE Error is:")
print(errtrain)


errtest = MSE(theta, x_test, y_test)
print("test MSE Error is:")
print(errtest)

x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
y = (-1 / theta[2]) * (theta[1] * x + theta[0])

plt.plot(x, y, color = "purple", label = "Decision_Boundary")
plt.legend()
plt.show()
# print(theta)
# print(predict(X))
