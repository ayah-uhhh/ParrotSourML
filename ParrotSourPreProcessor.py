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

from PSUtils import ANSWER_FILE, IMAGE_DIR, OUT_DIR, get_label, load_data


def write_img(start_positions, n):
    """
    Plot and output a start_position image for the Nth picture
    Parameters
    ----------
    start_positions : dict[]
        An array of dict {x,y} that contains all start positions for every group
        in every picture
    n : int
        The picture for which to generate an image
    """

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

    # Clean and prepare output directories
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    """ IMPORT DATA """
    loaded_data = load_data()

    """Extract numerical information"""
    Y = np.transpose(np.array([]))

    startPositions = {}

    n = 0

    print("Generating answer key and dataset...")
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
        Y = np.append(Y, get_label(k))
        n += 1

    # Save the answer key
    open(ANSWER_FILE, "w+")
    np.savetxt(ANSWER_FILE, Y, fmt="%s")

    # Create scatter plots for each n_group
    mpl.use("Agg")

    # Create a pool of threads, up to the # of cores avaialble,
    # to write image files
    pool = mp.Pool(mp.cpu_count())

    # Create images for each data row in the dataset
    results = [pool.apply_async(write_img, args=([startPositions, n]))
               for n in range(len(startPositions))]
    pool.close()

    # Required to track progress and get results
    for job in tqdm(results):
        job.get()

    total_time = time.time() - starttime
    print("Total time: " + str(total_time))

    with open("start_positions.json", "w") as file:  # save data
        json.dump(startPositions, file)
