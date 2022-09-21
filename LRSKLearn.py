





import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
from pandas import read_csv
import seaborn as sns
from sklearn.model_selection import train_test_split

#boston house price prediction
# Read and preprocessing Data
hour = pd.read_csv('hour.csv').dropna()
print(hour.info())

#changing the yyyy-mm-dd format to dd
list_dd=[]
list2=[]
for i in hour['dteday']:
    list1 = i.split('-')
    list_dd.append(int(list1[2]))
    list2.append(1)

dfh = pd.DataFrame(list_dd, columns=['dteday'])
hour[['dteday']]=dfh[['dteday']]
bias = pd.DataFrame(list2, columns=['bias'])
hour[['instant']]=bias[['bias']]


X = hour.iloc[:,0:-1]
Y = hour.iloc[:,-1]
# print(X.tail())
# print(np.shape(X))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)




from sklearn.linear_model import LinearRegression
#fitting our model to train and test
lm = LinearRegression()
model = lm.fit(x_train,y_train)


pred_y = lm.predict(x_test)

print(pd.DataFrame({"Actual": y_test, "Predict": pred_y}).head())
plt.scatter(y_test,pred_y)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title("Linear regression with SKLearn")
plt.show()

import sklearn
mse = sklearn.metrics.mean_squared_error(y_test, pred_y)
print(mse)


