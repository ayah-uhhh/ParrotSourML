# -*- coding: utf-8 -*-
"""
Created on 10 Jun 2023
@author: ayaha

This file runs a single RandomForest prediction with the coded img_size parameter.
Intended to be run with the best img_size found via the Pooled RF.
"""
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
from PSUtils import OUT_DIR, get_pics

psLog.setLevel(logging.DEBUG)

if __name__ == "__main__":
    mp.freeze_support()

    total_time = time.time()

    psLog.debug("Generating new images...")
    start_time = time.time()
    preprocess('data5000.json', 'predict')
    psLog.debug("Generated images. (%.2f)", time.time()-start_time)

    psLog.debug("Loading model...")
    start_time = time.time()
    loaded_model, loaded_model_settings = joblib.load(
        open('PSSVMSaved.jbl', 'rb'))
    clf = loaded_model()
    kernel, sea, shape, size_img = loaded_model_settings

    psLog.debug("Loaded model. (%.2fs)", time.time()-start_time)
    psLog.debug("  - Kernel: %s", kernel)
    psLog.debug("  - C: %s", sea)
    psLog.debug("  - Shape: %s", shape)
    psLog.debug("  - Img Size: %s", size_img)

    # Data must be read after loading model due to reliance on img_size from previous saved
    # model. In the event of a pooled run, we need to capture the config from the saved file
    # since we cannot guarantee the same result for each run
    psLog.debug("Reading data...")
    start_time = time.time()
    Y = np.loadtxt(os.path.join("predict", "Y.txt"), dtype=str)
    X = get_pics(size_img, os.path.join('predict', 'images'))
    psLog.debug("Loaded data (%.2fs)", time.time()-start_time)

    psLog.debug("Classifying...")
    start_time = time.time()
    predicted = clf.predict(X)

    elapsed_time = time.time() - start_time
    error_rate = 1 - metrics.accuracy_score(Y, predicted)

    psLog.debug("Classification complete. (%.2f)", elapsed_time)
    psLog.debug("Classification error: %.2f%%", (error_rate*100))

    # mpl.use("TkAgg")
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(
    #     Y, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # plt.show()
