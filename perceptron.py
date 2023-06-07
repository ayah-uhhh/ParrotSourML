# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:52:52 2023

@author: ayaha
"""

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def fit(self, X, y):
        # Add bias to input data
        X = np.insert(X, 0, 1, axis=1)
        
        # Initialize weights to zeros
        #self.weights = np.zeros(X.shape[1])
        self.weights = np.ones(X.shape[1])
        # Iterate through epochs
        for epoch in range(self.num_epochs):
            # Iterate through each training example
            for i in range(X.shape[0]):
                # Compute dot product of input and weights
                z = np.dot(X[i], self.weights)
                
                # Compute predicted output (1 if z >= 0, -1 otherwise)
                y_pred = np.where(z >= 0, 1, -1)
                
                # Update weights if prediction is incorrect
                if y[i] != y_pred:
                    self.weights += self.learning_rate * y[i] * X[i]
    
    def predict(self, X):
        # Add bias to input data
        X = np.insert(X, 0, 1, axis=1)
        
        # Compute dot product of input and weights
        z = np.dot(X, self.weights)
        
        # Compute predicted output (1 if z >= 0, -1 otherwise)
        y_pred = np.where(z >= 0, 1, -1)
        
        return y_pred