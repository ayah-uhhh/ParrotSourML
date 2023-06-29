# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha

Runs multi-threaded RandomForest predictions to optimize the img_size parameter.
"""
import logging
import multiprocessing as mp
import sys
import time

import joblib
from tqdm import tqdm

import PSSVM as psvm
from PSLogger import psLog

psLog.setLevel(logging.INFO)


def svm_pool(minsize=10, maxsize=20):
    if __name__ == '__main__':
        # required conditional to avoid recursive multithreading

        psLog.info("Finding optimized parameters...")
        psLog.debug("---------------")
        psLog.debug("Using image sizes: %s - %s", minsize, maxsize)
        psLog.info("---------------")

        starttime = time.time()

        # initialize a thread pool equal to the number of cores available on
        # this machine
        pool = mp.Pool(mp.cpu_count(), maxtasksperchild=2)

        # Start with a error rate of 100%; if a RF instance beats this
        # they become the new best. If an img_size of -1 is the best result,
        # something went wrong
        least_error = 100
        best_img_size = -1

        # Create threads for different img_size values
        # for x in range (10,50) will go through img_size values between 10 and 49
        # and find the best number within that range
        results = []

        # x is "C" value for SVM
        for size in range(minsize, maxsize):
            results.extend([pool.apply_async(psvm.psSVM, args=([False, "linear", x, 'ovr', size]))
                            for x in (10**i for i in range(-2, 4))])
            results.extend([pool.apply_async(psvm.psSVM, args=([False, "rbf", x, 'ovr', size]))
                            for x in (10**i for i in range(-2, 4))])

        pool.close()

        # get the results after each thread has finished executing
        output = []
        for job in tqdm(results):
            output.append(job.get())

        # find the best img_size (lowest error rate)
        for x in output:
            if (x[2] < least_error):
                least_error = x[2]
                best_img_size = x[0]
                model = x[3]

            # only in logLevel DEBUG, print all results
            psLog.debug("-----------")
            psLog.debug("Img size: %s", str(x[0]))
            psLog.debug("Error: %s", str(x[2]))
            psLog.debug("")
            psLog.debug("Img size: %s", str(x[1]))

        psLog.debug("Saving best svn model...")
        joblib.dump(model, 'PSSVMSaved.jbl')
        psLog.debug("Model saved.")
        psLog.info("------------------------------")

        total_time = time.time() - starttime

        # Results:
        psLog.info("Best error rate: %s", least_error)
        psLog.info("Best parameters: %s", best_img_size)
        psLog.info("Total time: %s", total_time)


##
#
# CLI arg 1 = img_size_min: minimum img-size to try
# CLI arg 2 = img_size max: maximum img-size to try
#
##
minsize = -1
maxsize = -1

if (len(sys.argv) > 1 and sys.argv[1].isnumeric()):
    minsize = int(sys.argv[1])

if (len(sys.argv) > 2 and sys.argv[2].isnumeric()):
    maxsize = int(sys.argv[2])

if (minsize == -1 and maxsize != -1):
    minsize = maxsize - 10
if (maxsize == -1 and minsize != -1):
    maxsize = minsize + 10

if (minsize == -1 and maxsize == -1):
    svm_pool()
else:
    svm_pool(minsize, maxsize)
