# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:23:34 2023
@author: ayaha
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

from PSLogger import psLog
from PSUtils import IMAGE_DIR, OUT_DIR, get_pics


def psSVM( save=False, kernel="linear", sea=1, shape='ovr', size_img=100, show_cm=False):
    """
    Higher C decreases the amount of misclassified data points in the trainng set
    but may increase misclassification in test data. C is log
    """

    start_time = time.time()
    psLog.debug("Loading data...")
    Y = np.loadtxt(OUT_DIR+'\\Y.txt', dtype=str)
    X = get_pics(size_img)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)
    psLog.debug("Loaded data (%s)", (time.time()-start_time))

    clf = svm.SVC(kernel=kernel, C=sea, decision_function_shape=shape)
    psLog.debug("Training model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    psLog.debug("Model trained (%s)", (time.time()-start_time))

    psLog.debug("Verifying accuracy...")
    start_time = time.time()
    predicted = clf.predict(X_test)
    error_rate = 1 - metrics.accuracy_score(y_test, predicted)
    elapsed_time = time.time() - start_time

    psLog.debug("Time taken to classify: %s seconds", elapsed_time)
    psLog.debug(f"Classification error: {error_rate}")

    psLog.debug("Saving model...")
    start_time = time.time()
    if save:
        model_settings = [kernel, sea, shape, size_img]
        joblib.dump((clf, model_settings), 'PSSVMSaved.jbl')
    psLog.debug("Model trained (%s)", (time.time()-start_time))

    if show_cm:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        plt.show()

    return [[kernel, sea, shape, size_img], elapsed_time, error_rate, (clf, model_settings)]
