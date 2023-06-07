# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 11:22:54 2023

@author: ayaha
"""
from numpy import *

# class LinearRegression:

#     def __init__(self, learning_rate=0.001, n_iters=1000):
#         self.lr = learning_rate
#         self.n_iters = n_iters
#         self.weights = None
#         self.bias = None

#     def fit(self, X, y):
#         (n_samples, n_features) = X.shape

#         # init parameters
#         self.weights = zeros(n_features)
#         self.bias = 1

#         # gradient descent
#         for _ in range(self.n_iters):
#             y_predicted = dot(X, self.weights) + self.bias
#             # compute gradients
#             dw = (1 / n_samples) * dot(X.T, (y_predicted - y))
#             db = (1 / n_samples) * sum(y_predicted - y)

#             # update parameters
#             self.weights -= self.lr * dw
#             self.bias -= self.lr * db


#     def predict(self, X):
       
#         y_approximated = dot(X, self.weights) + self.bias
#         return y_approximated

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        (n_samples,n_features) = X.shape
        # init parameters
        self.weights = zeros(n_features)
        self.bias = 1

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = dot(X,self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
       
        y_approximated = dot(X,self.weights) + self.bias
        return y_approximated