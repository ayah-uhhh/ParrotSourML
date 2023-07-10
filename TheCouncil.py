import logging
import multiprocessing as mp
import os
import time

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from ParrotSourPreProcessor import preprocess
from PSLogger import psLog
from PSUtils import get_pics

psLog.setLevel(logging.DEBUG)


def svm_test():
    total_time = time.time()

    psLog.debug("Preprocessing test data...")
    start_time = time.time()
    preprocess('data50.json', 'predict')
    psLog.debug('Test data preprocessed. (%.2f)', time.time()-start_time)

    psLog.debug("Loading SVM model...")
    start_time = time.time()
    loaded_svm, svm_settings = joblib.load(open('PSSVMSaved.jbl', 'rb'))
    clf = loaded_svm
    kernel, sea, shape, size_img = svm_settings
    psLog.debug("Loaded SVM model. (%.2fs)", time.time()-start_time)

    # Data must be read after loading model due to reliance on img_size from previous saved
    # model. In the event of a pooled run, we need to capture the config from the saved file
    # since we cannot guarantee the same result for each run
    psLog.debug("Reading data...")
    start_time = time.time()
    Y = np.loadtxt(os.path.join("predict", "Y.txt"), dtype=str)
    X = get_pics(size_img, os.path.join('predict', 'images'))
    psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

    psLog.debug("Classifying using SVM...")
    start_time = time.time()
    predicted_svm = clf.predict_proba(X)
    elapsed_time_svm = time.time() - start_time

    # Calculate the error rate by finding the index of the predicted class with the highest probability
    predicted_svm_labels = np.argmax(predicted_svm, axis=1)
    error_rate_svm = 1 - metrics.accuracy_score(Y, predicted_svm_labels)

    psLog.debug("SVM Classification complete. (%.2fs)",
                elapsed_time_svm)
    psLog.debug("SVM Classification error: %.2f%%",
                (error_rate_svm * 100))

    return predicted_svm, elapsed_time_svm, error_rate_svm


def random_forest_test():
    total_time = time.time()

    psLog.debug("Generating new images...")
    start_time = time.time()
    preprocess('data50.json', 'predict')
    psLog.debug("Generated images. (%.2f)", time.time()-start_time)

    psLog.debug("Loading Random Forest model...")
    start_time = time.time()
    loaded_rf, img_size = joblib.load(open('PSRandomForestSaved.jbl', 'rb'))
    forest = loaded_rf
    psLog.debug("Loaded Random Forest model. (%.2fs)", time.time()-start_time)

    psLog.debug("Reading data...")
    start_time = time.time()
    Y = np.loadtxt(os.path.join("predict", "Y.txt"), dtype=str)
    X = get_pics(img_size, os.path.join('predict', 'images'))
    psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

    psLog.debug("Classifying using Random Forest...")
    start_time = time.time()
    predicted_rf = forest.predict_proba(X)
    elapsed_time_rf = time.time() - start_time

    # Calculate the error rate by finding the index of the predicted class with the highest probability
    predicted_rf_labels = np.argmax(predicted_rf, axis=1)
    error_rate_rf = 1 - metrics.accuracy_score(Y, predicted_rf_labels)

    psLog.debug("Random Forest Classification complete. (%.2fs)",
                elapsed_time_rf)
    psLog.debug("Random Forest Classification error: %.2f%%",
                (error_rate_rf * 100))

    return predicted_rf, elapsed_time_rf, error_rate_rf


if __name__ == '__main__':
    mp.freeze_support()

    predicted_svm, elapsed_time_svm, error_rate_svm = svm_test()
    predicted_rf, elapsed_time_rf, error_rate_rf = random_forest_test()

    labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2,
                'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}

    # Load 'Y' from the test data
    Y = np.loadtxt(os.path.join("predict", "Y.txt"), dtype=str)

    # Convert string labels to numerical values
    Y = np.array([labelmap[label] for label in Y])

    # Calculate weighted average
    weight_svm = 0.6  # Arbitrary
    weight_rf = 0.4

    weighted_predicted = weight_svm * predicted_svm + weight_rf * predicted_rf
    final_predicted = np.argmax(weighted_predicted, axis=1)

    # Convert numerical labels back to string labels
    labelmap_inverse = {v: k for k, v in labelmap.items()}
    final_predicted_labels = np.array([labelmap_inverse[label]
                                       for label in final_predicted])

    weighted_error_rate = 1 - metrics.accuracy_score(Y, final_predicted)

    # Debug logging
    psLog.debug("Weighted Average Classification complete.")
    psLog.debug("Final Predicted Labels: %s", final_predicted_labels)
    psLog.debug("True Labels: %s", Y)
    psLog.debug("Weighted Average Classification error: %.2f%%",
                (weighted_error_rate * 100))
