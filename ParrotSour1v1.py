# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:23:34 2023

@author: ayaha
"""
import time
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from PIL import Image
import os
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
# import cv2

start_time = time.time()

pictures = []
for root, dirs, files in os.walk("./output", topdown=True):
    filelist = []
    for n in files:
        if ("Y.txt" not in n):
            filelist.append(os.path.splitext(n)[0])
    sorted_files = sorted(filelist, key=int)

    for name in sorted_files:
        if ("Y.txt" not in name):
            image = Image.open(os.path.join(root, name)+'.png')
            resized_image = image.resize((100, 100))
            image_array = np.array(resized_image).flatten()
            pictures.append(image_array)
"""Higher C decreases the amount of misclassified data points in the trainng set
but may increase misclassification in test data. C is log

"""
clf = svm.SVC(kernel="linear", C=1, decision_function_shape='ovo')
Y = np.loadtxt('output\\Y.txt', dtype=str)
X = np.asarray(pictures)
# Print the loaded array

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

error = 1 - metrics.accuracy_score(y_test, predicted)
print(f"Classification error: {error}")
end_time = time.time()
elapsed_time = end_time - start_time

print("Time taken to classify:", elapsed_time, "seconds")

debug = True
if (debug):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
