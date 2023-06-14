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


def psSVM(save=False, kernel="linear", sea=1, shape='ovr', size_img=100, show_cm=False):
    """
    Higher C decreases the amount of misclassified data points in the trainng set
    but may increase misclassification in test data. C is log
    """
    total_time = time.time()

    start_time = time.time()
    psLog.debug("Loading data...")
    Y = np.loadtxt(OUT_DIR+'\\Y.txt', dtype=str)
    X = get_pics(size_img)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)
    psLog.debug("Loaded data (%.2fs)", (time.time()-start_time))

    clf = svm.SVC(kernel=kernel, C=sea, decision_function_shape=shape)
    psLog.debug("Training model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    psLog.debug("Model trained (%.2fs)", (time.time()-start_time))

    psLog.debug("Verifying accuracy...")
    start_time = time.time()
    predicted = clf.predict(X_test)
    error_rate = 1 - metrics.accuracy_score(y_test, predicted)
    elapsed_time = time.time() - start_time

    psLog.debug("Classification complete. (%.2fs)", elapsed_time)
    psLog.debug(f"Classification error: {error_rate}")

    model_settings = [kernel, sea, shape, size_img]
    if save:
        psLog.debug("Saving model...")
        start_time = time.time()
        joblib.dump((clf, model_settings), 'PSSVMSaved.jbl')
        psLog.debug("Model saved (%.2fs)", (time.time()-start_time))

    elapsed_time = time.time() - total_time
    psLog.debug("Total time: %.2fs", time.time()-total_time)

    if show_cm:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        plt.show()

    return [model_settings, elapsed_time, error_rate, (clf, model_settings)]
