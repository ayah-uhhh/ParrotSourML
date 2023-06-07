# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:23:34 2023

@author: ayaha
"""
import numpy as np 
import matplotlib.pyplot as plt

"""
 Labels
 1: Azimuth, 2: Range, 3: Wall
 4: Ladder,  5: Champagne 6: Vic
 7: Single
 svm where it is 1 vs many
 sklearn confidences 
 
 chapter challenges
 background
 data set bias
 pre-processing
 2d data
 multi class vs binary
 variance of a ml algorithm(svms)
 results(confusion matrix) possibly generate new features 
 
 """
import os
#import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import time

start_time = time.time()
pathtoimages = '\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000'
os.chdir(pathtoimages)

pictures = []
for n in range (0,1000):
    filename = f"group_{n}.png"
    image = Image.open(filename)
    resized_image = image.resize((100,100))
    image_array = np.array(resized_image).flatten()
    pictures.append(image_array)
"""Higher C decreases the amount of misclassified data points in the trainng set
but may increase misclassification in test data. C is log

"""
clf = svm.SVC(kernel="linear", C=1, decision_function_shape='ovo')
Y = np.loadtxt('\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000\\Y.txt',dtype=str)
X = np.asarray(pictures)
# Print the loaded array

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

error = 1 - metrics.accuracy_score(y_test, predicted)
print(f"Classification error: {error}")
end_time = time.time()
elapsed_time = end_time - start_time

print("Time taken to classify:", elapsed_time, "seconds")
# import os
# import numpy as np
# from PIL import Image

# # Set the path to the directory containing the images
# pathtoimages = '.'

# # Change the working directory to the path to the images
# os.chdir(pathtoimages)

# # Define a function to load an image and convert it to a flattened grayscale array
# def load_image(filename):
#     image = Image.open(filename)
#     image_array = np.array(image.convert('L')).flatten()
#     return image_array

# # Load the labels from the Y.txt file
# with open('Y.txt') as f:
#     Y = np.array(f.read().splitlines())

# # Load the images and convert them to flattened grayscale arrays
# pictures = []
# for n in range(1000):
#     filename = f'group_{n}.png'
#     image_array = load_image(filename)
#     pictures.append(image_array)

# Convert the pictures list to a numpy array
#X = np.asarray(pictures)

# Define a function to compute the RBF kernel
# def rbf_kernel(X, Y, gamma=0.1):
#     """Compute the RBF kernel between two matrices X and Y."""
#     pairwise_dists = -2 * np.dot(X, Y.T) + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
#     kernel = np.exp(-gamma * pairwise_dists)
#     return kernel

# # Train an SVM classifier with an RBF kernel
# C = 1.0  # regularization parameter
# gamma = 0.1  # kernel parameter
# n_samples = X.shape[0]
# K = rbf_kernel(X, X, gamma=gamma)
# alphas = np.zeros(n_samples)
# b = 0
# tolerance = 1e-4
# max_iter = 100

# for iteration in range(max_iter):
#     num_changed_alphas = 0
#     for i in range(n_samples):
#         E_i = b + np.sum(alphas * Y * K[:, i]) - Y[i]
#         if (Y[i] * E_i < -tolerance and alphas[i] < C) or (Y[i] * E_i > tolerance and alphas[i] > 0):
#             j = np.random.choice([k for k in range(n_samples) if k != i])
#             E_j = b + np.sum(alphas * Y * K[:, j]) - Y[j]
#             alpha_i_old, alpha_j_old = alphas[i], alphas[j]
#             if Y[i] == Y[j]:
#                 L = max(0, alphas[j] + alphas[i] - C)
#                 H = min(C, alphas[j] + alphas[i])
#             else:
#                 L = max(0, alphas[j] - alphas[i])
#                 H = min(C, C + alphas[j] - alphas[i])
#             if L == H:
#                 continue
#             eta = 2 * K[i, j] - K[i, i] - K[j, j]
#             if eta >= 0:
#                 continue
#             alphas[j] -= Y[j] * (E_i - E_j) / eta
#             alphas[j]
