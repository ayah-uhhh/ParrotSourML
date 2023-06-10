# -*- coding: utf-8 -*-

"""

Created on Sun Apr 23 08:46:39 2023


@author: ayaha

"""
import PSRandomForest as psrf
import time
from tqdm import tqdm

debug = False

if __name__ == '__main__':

    import multiprocessing as mp

    starttime = time.time()

    pool = mp.Pool(mp.cpu_count())

    print("Finding optimized parameters...")
    print("---------------")

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
        if (debug):
            print("-----------")
            print("Img size: " + str(x[0]))
            print("Error: " + str(x[2]))
            print("")
            print("Img size: " + str(x[1]))
        if (x[2] < least_error):
            least_error = x[2]
            best_img_size = x[0]

    print("------------------------------")

    total_time = time.time() - starttime

    print("Best error rate: ", least_error)
    print("Best img size: ", best_img_size)
    print("Total time: " + str(total_time))
