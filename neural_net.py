# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 00:14:49 2018

@author: Illusion'Ic
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#perceptron initialization
class Perceptron(object):

    def __init__(self, eta=0.01, epochs=200):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, -1, 1)

#setosa class from the dataset
        
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# sepal length and petal length
X = df.iloc[0:100, [0,2]].values

ppn = Perceptron(epochs=200, eta=0.01)
ppn.train(X, y)

#adaline stocastic gradient descent class
class AdalineSGD(object):

    def __init__(self, eta=0.01, epochs=200):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, -1, 1)
    
 #standardizing features
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#sampling with replacement 
ada = AdalineSGD(epochs=200, eta=0.01)

#shuffling data at random
np.random.seed(250)
idx = np.random.permutation(len(y))
X_shuffled, y_shuffled =  X_std[idx], y[idx]

# training through adaline and plotting decision regions
ada.train(X_shuffled, y_shuffled)

plot_decision_regions(X_shuffled, y_shuffled, clf=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal length [after_standardization]')
plt.ylabel('Petal length [after_standardization]')
plt.show()
#plotting the loss against 200 ephocs
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum of squared-error')
plt.show()

#printing the values of our thetas
print('Weights: %s' % ppn.w_)
