# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha

Defines a function to run a single RandomForest prediction with the coded img_size parameter.
"""
import logging
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from PSLogger import psLog
from PSUtils import IMAGE_DIR, OUT_DIR, get_pics


def randomforest(save=False, img_size=30, n_estimators=240, use_pca=False, show_cm=False):
    """
    Run sklearn random forest model training and prediction.

    This function will take an output image, bin the pixels into img_size sized
    bins, and use sklearn's Random Forest to train and predict.

    Parameters
    ----------
    img_size : int (optional)
        The size (width & height) in which to bin the larger image
    n_estimators : int (optional)
        The number of trees in the forest
    use_pca : bool (optional)
        Whether or not to use PCA optimization
    show_cm : bool (optional)
        Whether or not to display the confusion matrix after running algorithm

    Returns
    ----------
    img_size : int 
        The image size used for binning images in this RF run
    elapsed_time : int
        How much time the algorithm took to run for this img_size
    error_rate : int 
        The rate of error in classifying the test data
    """
    total_time = time.time()

    psLog.debug("Loading data...")
    start_time = time.time()
    # load the answer key and format the dataset
    Y = np.loadtxt(OUT_DIR+"\\Y.txt", dtype=str)
    X = get_pics(img_size)

    # run optimization if specified
    if (use_pca):
        psLog.debug("Using PCA optimization...")
        num_components = len(X) if len(X) < 7 else 7
        pca = PCA(n_components=num_components)
        X = pca.fit_transform(X)

    # train the model
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)
    psLog.debug("Loaded data (%.2fs)", (time.time()-start_time))

    # create the rf classifier
    start_time = time.time()
    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest.fit(x_train, y_train)
    psLog.debug("Model trained (%.2fs)", (time.time()-start_time))

    psLog.debug("Verifying accuracy...")
    start_time = time.time()
    # predict based on test data
    predicted = forest.predict(x_test)
    error_rate = 1 - metrics.accuracy_score(y_test, predicted)

    psLog.debug("Classification complete. (%.2fs)", time.time()-start_time)
    psLog.debug("Classification error: %.2f%%", (error_rate*100))

    if save:
        psLog.debug("Saving model...")
        start_time = time.time()
        joblib.dump(forest, 'PSRandomForestSaved.jbl')
        psLog.debug("Model saved (%.2fs)", (time.time()-start_time))

    elapsed_time = time.time()-total_time
    psLog.debug("Total time: %.2fs", elapsed_time)

    if show_cm:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        plt.show()

    return [img_size, elapsed_time, error_rate, forest]
