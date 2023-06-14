# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:47:31 2023

@author: ayaha
"""
import json
import multiprocessing as mp
import os
import shutil
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PSLogger import psLog
from PSUtils import ANSWER_FILE, IMAGE_DIR, OUT_DIR, get_label, load_data


def write_img(start_positions, n, outdir=OUT_DIR):
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
    file_path = os.path.join(outdir, "images", file_name)
    plt.savefig(file_path)
    plt.close(fig)


def clean_dirs(outdir):
    # Clean and prepare output directories
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir, "images")):
        os.makedirs(os.path.join(outdir, "images"))


def preprocess(filename="data1000.json", outdir=OUT_DIR):
    mp.freeze_support()

    total_time = time.time()

    clean_dirs(outdir)

    """ IMPORT DATA """
    psLog.debug("Loading PS data...")
    start_time = time.time()
    loaded_data = load_data(filename)

    """Extract numerical information"""
    Y = np.transpose(np.array([]))

    start_positions = {}

    n = 0

    psLog.info("Generating answer key and dataset...")
    for k in loaded_data:  # pictures
        groups = k.get("groups")
        start_positions[n] = {}
        newx = []
        newy = []
        for i in range(len(groups)):  # group number in pictures
            # individual points in groups
            for j in range(0, len(groups[i])):
                start_pos = groups[i][j].get("startPos")
                newx.append(start_pos.get("x"))
                newy.append(start_pos.get("y"))
            start_positions[n]["x"] = newx
            start_positions[n]["y"] = newy

        # Make the answer key
        Y = np.append(Y, get_label(k))
        n += 1

    psLog.debug('Loaded data. (%.2f)', time.time()-start_time)

    # Save the answer key
    open(os.path.join(outdir, "Y.txt"), "w+")
    np.savetxt(os.path.join(outdir, "Y.txt"), Y, fmt="%s")

    # Create scatter plots for each n_group
    mpl.use("Agg")

    # Create a pool of threads, up to the # of cores avaialble,
    # to write image files
    pool = mp.Pool(mp.cpu_count())

    psLog.debug("Generating images...")
    start_time = time.time()
    # Create images for each data row in the dataset
    results = [pool.apply_async(write_img, args=([start_positions, n, outdir]))
               for n in range(len(start_positions))]
    pool.close()

    # Required to track progress and get results
    for job in tqdm(results):
        job.get()

    psLog.debug("Generated images. (%.2f)", time.time()-start_time)

    psLog.info("Total time: %.2f", (time.time()-total_time))

    with open("start_positions.json", "w") as file:  # save data
        json.dump(start_positions, file)


if __name__ == "__main__":
    mp.freeze_support()
    preprocess("data1000.json")
