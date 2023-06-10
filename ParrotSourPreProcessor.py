# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:47:31 2023

@author: ayaha
"""
from PIL import Image
from alive_progress import alive_bar
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import csv
import glob
import time
from tqdm import tqdm


def write_file(startPositions, n):

    # img_size = (500, 500)
    save_dir = "output"

    fig, ax = plt.subplots()
    ax.scatter(startPositions[n]["x"],
               startPositions[n]["y"], c="black", marker="3")
    ax.set_title("")  # (f"Group {n}")
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
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    plt.close(fig)

    # with Image.open(file_path) as img:
    #     center_x = img.width // 2
    #     center_y = img.height // 2
    #     left = max(0, center_x - img_size[0] // 2)
    #     right = min(img.width, center_x + img_size[0] // 2)
    #     top = max(0, center_y - img_size[1] // 2)
    #     bottom = min(img.height, center_y + img_size[1] // 2)
    #     cropped_img = img.crop((left, top, right, bottom))
    #     cropped_img.save(file_path)


if __name__ == '__main__':

    import multiprocessing as mp

    starttime = time.time()

    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

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
    file_path = os.path.join("output", "Y.txt")

    if not os.path.exists("output"):
        os.makedirs("output")

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
