# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha

Tests CNN with 5000 images.
"""
import logging
import multiprocessing as mp
import os
import time

import tensorflow as tf

from ParrotSourPreProcessor import preprocess
from PSCNNUtils import get_cnn_pics, one_hot_encode_labels
from PSLogger import psLog

psLog.setLevel(logging.DEBUG)

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

if __name__ == '__main__':
    mp.freeze_support()
    total_time = time.time()

    psLog.debug("Preprocessing test data...")
    start_time = time.time()
    preprocess('data5000.json', 'predict')
    psLog.debug('Test data preprocessed. (%.2f)', time.time()-start_time)

    psLog.debug("Loading model...")
    start_time = time.time()
    model = tf.keras.models.load_model('ps_cnn_model_2.h5')

    model.optimizer = best_optimizer
    model.filters = best_filters
    model.kernel_size = (best_kernel_size, best_kernel_size)
    model.img_size = best_img_size

    # Data must be read after loading model due to reliance on img_size from previous saved
    # model. In the event of a pooled run, we need to capture the config from the saved file
    # since we cannot guarantee the same result for each run
    psLog.debug("Reading data...")
    start_time = time.time()
    Y = one_hot_encode_labels(os.path.join('predict', "Y.txt"))
    X = get_cnn_pics(os.path.join('predict', 'images'))

    psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

    psLog.debug("Classifying...")
    start_time = time.time()
    loss, accuracy = model.evaluate(X, Y, verbose=0)
    elapsed_time = time.time() - start_time

    psLog.debug("Classification complete. (%.2fs)", elapsed_time)
    psLog.debug("Classification accuracy: %.2f%%", (accuracy*100))
