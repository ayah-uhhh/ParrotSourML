# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha

Defines a function to run a single RandomForest prediction with the coded img_size parameter.
"""
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import tensorflow as tf

from ParrotSourPreProcessor import preprocess
from PSCNNUtils import decode_label, get_cnn_pic, one_hot_encode_labels
from PSLogger import psLog

psLog.setLevel(logging.INFO)

if (len(sys.argv) != 2):
    psLog.warn("Must provide PS data file")
    exit()

single_image = sys.argv[1]

if not os.path.exists(os.path.join('trainingdata', single_image)):
    psLog.warn("File does not exist.")
    exit()

##
# TODO - convert this to a function that takes the file as a parammeter
##

##
# TODO - convert to accept JSON object instead (or additional function?)
##

##
# TODO - investigate how to accelerate preprocessing time (9s to make one prediction is too slow)
##

if __name__ == '__main__':
    mp.freeze_support()
    total_time = time.time()

    psLog.debug("Preprocessing...")
    preprocess(single_image, 'predictsingle')
    with open(os.path.join('predictsingle', 'Y.txt'), "a") as myfile:
        myfile.write("SINGLE")

    psLog.debug("Loading model...")
    start_time = time.time()
    model = tf.keras.models.load_model('ps_cnn_model.h5')

    # Data must be read after loading model due to reliance on img_size from previous saved
    # model. In the event of a pooled run, we need to capture the config from the saved file
    # since we cannot guarantee the same result for each run
    psLog.debug("Reading data...")
    start_time = time.time()
    Y = [one_hot_encode_labels(os.path.join('predictsingle', "Y.txt"))[0]]
    Y = np.asarray(Y)
    X = get_cnn_pic(os.path.join('predictsingle', 'images', '0.png'))
    psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

    psLog.debug("Classifying...")
    start_time = time.time()
    predictions = model.predict(X, verbose=0)
    elapsed_time = time.time() - start_time

    predicted_label = np.argmax(predictions)
    psLog.debug("Actual label: %s", Y)

    psLog.info("Predicted: %s", decode_label(predicted_label))
    psLog.info("Time: %.2f", time.time()-total_time)
    psLog.debug("Actual label: %s", decode_label(Y[0]))
    if (predicted_label == Y[0]):
        psLog.info("Correct.")
    else:
        psLog.info('Incorrect')

    psLog.debug("Classification complete. (%.2fs)", elapsed_time)
