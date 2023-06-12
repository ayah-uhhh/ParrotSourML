# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:23:34 2023
@author: ayaha
"""
import os
import time
from PSLogger import psLog
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

from PSUtils import IMAGE_DIR, OUT_DIR, get_pics


def psSVM(kernel="linear", sea=1, shape='ovr', size_img=100, show_cm=False,):

    start_time = time.time()

    """
    Higher C decreases the amount of misclassified data points in the trainng set
    but may increase misclassification in test data. C is log
    """

    clf = svm.SVC(kernel=kernel, C=sea, decision_function_shape=shape)
    Y = np.loadtxt(OUT_DIR+'\\Y.txt', dtype=str)
    X = get_pics(size_img)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    error_rate = 1 - metrics.accuracy_score(y_test, predicted)
    end_time = time.time()
    elapsed_time = end_time - start_time

    psLog.debug("Time taken to classify: %s seconds", elapsed_time)
    psLog.debug(f"Classification error: {error_rate}")

    if show_cm:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        plt.show()
    return [[kernel, sea, shape, size_img], elapsed_time, error_rate]
