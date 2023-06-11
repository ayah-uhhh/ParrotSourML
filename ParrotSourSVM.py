# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:23:34 2023
@author: ayaha
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

from ParrotSourPreProcessor import IMAGE_DIR, OUT_DIR

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
debug = True

start_time = time.time()

pictures = []
filelist = []
for root, dirs, files in os.walk(IMAGE_DIR, topdown=True):

    for n in files:
        filelist.append(os.path.splitext(n)[0])

sorted_files = sorted(filelist, key=int)


for name in sorted_files:
    image = Image.open(os.path.join(root, name)+'.png')
    resized_image = image.resize((100, 100))
    image_array = np.array(resized_image).flatten()
    pictures.append(image_array)

"""
Higher C decreases the amount of misclassified data points in the trainng set
but may increase misclassification in test data. C is log
"""
clf = svm.SVC(kernel="linear", C=1, decision_function_shape='ovo')
Y = np.loadtxt(OUT_DIR+'\\Y.txt', dtype=str)
X = np.asarray(pictures)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

error = 1 - metrics.accuracy_score(y_test, predicted)
print(f"Classification error: {error}")
end_time = time.time()
elapsed_time = end_time - start_time

print("Time taken to classify:", elapsed_time, "seconds")

if (debug):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
