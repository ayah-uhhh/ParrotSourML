"""tf trial"""
import logging
import os
import time
import multiprocessing as mp
# from tdqm import tdqm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split

from PSCNNUtils import get_cnn_pics, one_hot_encode_labels
from PSLogger import psLog
from PSUtils import IMAGE_DIR, OUT_DIR
import PSCNN as pscnn

psLog.setLevel(logging.DEBUG)


if __name__ == '__main__':
    # required conditional to avoid recursive multithreading

    psLog.info("Finding optimized parameters...")
    psLog.info("---------------")

    starttime = time.time()

    # initialize a thread pool equal to the number of cores available on
    # this machine
    pool = mp.Pool(mp.cpu_count(), maxtasksperchild=3)

    # Start with a error rate of 100%; if a RF instance beats this
    # they become the new best. If an img_size of -1 is the best result,
    # something went wrong
    least_error = 100
    best_img_size = -1

    # Create threads for different optimizers, filters, kernel sizes and img_size values
    # this will find the best number within each range
    optimizer = ['rmsprop', 'nadam', 'adam']
    for i in range(2):
        for j in range(1, 21):
            for l in range(10, 101, 10):
                results = [pool.apply_async(
                    pscnn.pscnn, args=([optimizer[i], j, (j, j), l]))]

    pool.close()

    # get the results after each thread has finished executing
    output = []
    for job in results:
        output.append(job.get())

    for x in output:
        if (x[2] < least_error):
            least_error = x[2]
            best_img_size = x[0]
            news = x[3]

        # only in logLevel DEBUG, print all results
        psLog.debug("-----------")
        psLog.debug("Img size: %s", str(x[0]))
        psLog.debug("Error: %s", str(x[2]))
        psLog.debug("")
        psLog.debug("Img size: %s", str(x[1]))

    psLog.debug("Saving best CNN model...")
    pscnn.model.save((news, best_img_size), 'ps_cnn_model_2.h5')
    psLog.debug("Model saved.")
    psLog.info("------------------------------")

    total_time = time.time() - starttime

    # Results:
    psLog.info("Best error rate: %s", least_error)
    psLog.info("Best img size: %s", best_img_size)
    psLog.info("Total time: %s", total_time)
