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
import numpy as np
import tensorflow as tf
from sklearn import metrics

from ParrotSourPreProcessor import preprocess
from PSLogger import psLog
from PSUtils import get_pics

psLog.setLevel(logging.DEBUG)

total_time = time.time()

psLog.debug("Generating new images....")
start_time = time.time()
preprocess('data50.json', 'predict')
psLog.debug('Generated images. (%.2f)', time.time()-start_time)

psLog.debug("Loading model...")
start_time = time.time()
model = tf.keras.models.load_model('my_model.h5')

# Data must be read after loading model due to reliance on img_size from previous saved
# model. In the event of a pooled run, we need to capture the config from the saved file
# since we cannot guarantee the same result for each run
psLog.debug("Reading data...")
start_time = time.time()
Y = np.loadtxt(os.path.join("predict", "Y.txt"), dtype=str)
X = get_pics(100, os.path.join('predict', 'images'))
psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

psLog.debug("Classifying...")
start_time = time.time()
loss, accuracy = model.evaluate(X, Y, verbose=0)
elapsed_time = time.time() - start_time

psLog.debug("Classification complete. (%.2fs)", elapsed_time)
psLog.debug("Classification accuracy: %.2f%%", (accuracy*100))
