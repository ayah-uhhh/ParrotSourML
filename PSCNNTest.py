# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha

Defines a function to run a single RandomForest prediction with the coded img_size parameter.
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

if __name__ == '__main__':
    mp.freeze_support()
    total_time = time.time()

    psLog.debug("Preprocessing test data...")
    start_time = time.time()
    preprocess('data50.json', 'predict')
    psLog.debug('Test data preprocessed. (%.2f)', time.time()-start_time)

    psLog.debug("Loading model...")
    start_time = time.time()
    model = tf.keras.models.load_model('ps_cnn_model.h5')

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
