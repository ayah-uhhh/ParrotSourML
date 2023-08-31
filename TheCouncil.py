from PSCNNUtils import get_cnn_pics, one_hot_encode_labels
import tensorflow as tf
from PSUtils import get_pics
from PSLogger import psLog
from ParrotSourPreProcessor import preprocess
from sklearn import metrics
import matplotlib.pyplot as plt
import logging
import multiprocessing as mp
import os
import time

import joblib
import numpy as np


psLog.setLevel(logging.DEBUG)

json_data = 'data50.json'


def svm_test():

    psLog.debug("Preprocessing test data...")
    start_time = time.time()
    preprocess(json_data, 'predict')
    psLog.debug('Test data preprocessed. (%.2f)', time.time()-start_time)

    psLog.debug("Loading SVM model...")
    start_time = time.time()
    loaded_svm, svm_settings = joblib.load(open('PSSVMSaved.jbl', 'rb'))
    clf = loaded_svm
    kernel, sea, shape, size_img = svm_settings
    psLog.debug("Loaded SVM model. (%.2fs)", time.time()-start_time)

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

    psLog.debug("Generating new images...")
    start_time = time.time()
    preprocess(json_data, 'predict')
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


def cnn_test():

    with open('best_params.txt', 'r') as file:
        best_params = file.read()
    best_params = best_params.replace(
        '(', '').replace(')', '').replace(',', '')
    best_params_list = best_params.split()

    # Extract the best parameters
    best_optimizer = best_params_list[0]
    best_filters = int(best_params_list[1])
    best_kernel_size = int(best_params_list[2])
    best_img_size = int(best_params_list[4])

    psLog.debug("Preprocessing test data...")
    start_time = time.time()
    preprocess(json_data, 'predict')
    psLog.debug('Test data preprocessed. (%.2f)', time.time()-start_time)

    psLog.debug("Loading model...")
    start_time = time.time()
    model = tf.keras.models.load_model('ps_cnn_model.h5')

    model.optimizer = best_optimizer
    model.filters = best_filters
    model.kernel_size = (best_kernel_size, best_kernel_size)
    model.img_size = best_img_size

    psLog.debug("Reading data...")
    start_time = time.time()
    Y = one_hot_encode_labels(os.path.join('predict', "Y.txt"))
    X = get_cnn_pics(os.path.join('predict', 'images'))

    psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

    psLog.debug("Classifying...")
    start_time = time.time()
    predicted_cnn = model.predict(X)
    elapsed_time_cnn = time.time() - start_time

    # Calculate the error rate by finding the index of the predicted class with the highest probability
    predicted_cnn_labels = np.argmax(predicted_cnn, axis=1)
    error_rate_cnn = 1 - \
        metrics.accuracy_score(Y, predicted_cnn_labels)

    psLog.debug("CNN Classification complete. (%.2fs)", elapsed_time_cnn)
    psLog.debug("CNN Classification error: %.2f%%", (error_rate_cnn * 100))

    return predicted_cnn, elapsed_time_cnn, error_rate_cnn


if __name__ == '__main__':
    mp.freeze_support()

    with mp.Pool(processes=3) as pool:
        predicted_svm, elapsed_time_svm, error_rate_svm = svm_test()
        predicted_rf, elapsed_time_rf, error_rate_rf = random_forest_test()
        predicted_cnn, elapsed_time_cnn, error_rate_cnn = cnn_test()

    # Store the error rates for each algorithm
    error_rates = {
        "SVM": error_rate_svm,
        "Random Forest": error_rate_rf,
        "CNN": error_rate_cnn
    }

    labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2,
                'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}

    # Load 'Y' from the test data
    Y = np.loadtxt(os.path.join("predict", "Y.txt"), dtype=str)

    # Convert string labels to numerical values
    Y = np.array([labelmap[label] for label in Y])

    # Calculate weighted average
    weight_svm = 0.3  # Arbitrary
    weight_rf = 0.3
    weight_cnn = 0.4

    weighted_predicted = (weight_svm * predicted_svm) + \
        (weight_rf * predicted_rf) + (weight_cnn * predicted_cnn)
    final_predicted = np.argmax(weighted_predicted, axis=1)

    # Convert numerical labels back to string labels
    labelmap_inverse = {v: k for k, v in labelmap.items()}
    final_predicted_labels = np.array([labelmap_inverse[label]
                                       for label in final_predicted])

    weighted_error_rate = 1 - metrics.accuracy_score(Y, final_predicted)

    # Convert numerical labels back to string labels
    labelmap_inverse = {v: k for k, v in labelmap.items()}
    final_true_labels = np.array([labelmap_inverse[label] for label in Y])

    # Debug logging
    psLog.debug("Weighted Average Classification complete.")
    psLog.debug("Final Predicted Labels: %s", final_predicted_labels)
    psLog.debug("True Labels: %s", final_true_labels)  # Print the class labels
    psLog.debug("Weighted Average Classification error: %.2f%%",
                (weighted_error_rate * 100))

    # Print the error rates of each algorithm
    for algo, error_rate in error_rates.items():
        print(f"{algo} Error Rate: {error_rate:.2%}")

    # Create a comparison graph
    plt.figure(figsize=(10, 6))
    plt.bar(error_rates.keys(), error_rates.values())
    plt.xlabel('Algorithms')
    plt.ylabel('Error Rate')
    plt.title('Algorithm Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save or display the graph
    plt.savefig('algorithm_comparison.png')
