# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:47:31 2023

@author: ayaha
"""
import csv
import glob
import json
import os
import shutil
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from PIL import Image
from tqdm import tqdm

OUT_DIR = "output"
IMAGE_DIR = OUT_DIR+"\\images"


def write_file(start_positions, n):

    fig, ax = plt.subplots()
    ax.scatter(start_positions[n]["x"],
               start_positions[n]["y"], c="black", marker="3")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(0, 400)
    ax.set_ylim(-200, 200)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    file_name = f"{n}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(file_path)
    plt.close(fig)


if __name__ == '__main__':

    import multiprocessing as mp

    starttime = time.time()

    shutil.rmtree(OUT_DIR)

    """ IMPORT DATA """
    data = open("trainingdata\data1000.json")
    loaded_data = json.load(data)

    """Extract numerical information"""
    Y = np.transpose(np.array([]))

    startPositions = {}

    labels = ['AZIMUTH', 'RANGE', 'WALL',
              "LADDER", "CHAMPAGNE", "VIC", "SINGLE"]
    n = 0

    print("Generating answer key...")
    for k in loaded_data:  # pictures
        groups = k.get("groups")
        startPositions[n] = {}
        newx = []
        newy = []
        for i in range(len(groups)):  # group number in pictures
            for j in range(0, len(groups[i])):  # individual points in groups
                startPos = groups[i][j].get("startPos")
                newx.append(startPos.get("x"))
                newy.append(startPos.get("y"))
            startPositions[n]["x"] = newx
            startPositions[n]["y"] = newy

        # Make the answer key
        if "pic" in k:
            val = k.get("pic")

            found_label = [label for label in labels if (label in val)][0]
            Y = np.append(Y, found_label)
        n += 1

    """
    Take X1 and X2 ---> matrix of points
    """
    """
    I want one plot for every set of points in the same group {n}
    """
    """ plot for ML """

    # Create scatter plots for each n_group
    file_path = os.path.join(OUT_DIR, "Y.txt")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    open(file_path, "w+")
    np.savetxt(file_path, Y, fmt="%s")

    mpl.use("Agg")

    pool = mp.Pool(mp.cpu_count()-1)
    results = [pool.apply_async(write_file, args=([startPositions, n]))
               for n in range(len(startPositions))]
    pool.close()
    output = []
    for job in tqdm(results):
        output.append(job.get())

    total_time = time.time() - starttime
    print("Total time: " + str(total_time))

    with open("start_positions.json", "w") as file:  # save data
        json.dump(startPositions, file)
