# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha
"""
import logging
import time

from tqdm import tqdm

import PSRandomForest as psrf
from PSLogger import psLog

psLog.setLevel(logging.INFO)

if __name__ == '__main__':

    import multiprocessing as mp

    starttime = time.time()

    pool = mp.Pool(mp.cpu_count())

    psLog.info("Finding optimized parameters...")
    psLog.info("---------------")

    least_error = 100

    best_img_size = -1

    results = [pool.apply_async(psrf.randomforest, args=([x]))
               for x in range(10, 50)]
    results.append(pool.apply_async(psrf.randomforest, args=([15])))

    pool.close()

    output = []
    for job in tqdm(results):
        output.append(job.get())

    for x in output:
        if (x[2] < least_error):
            least_error = x[2]
            best_img_size = x[0]

        psLog.debug("-----------")
        psLog.debug("Img size: %s", str(x[0]))
        psLog.debug("Error: %s", str(x[2]))
        psLog.debug("")
        psLog.debug("Img size: %s", str(x[1]))

    psLog.info("------------------------------")

    total_time = time.time() - starttime

    psLog.info("Best error rate: %s", least_error)
    psLog.info("Best img size: %s", best_img_size)
    psLog.info("Total time: %s", total_time)
