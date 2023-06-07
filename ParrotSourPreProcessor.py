# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:47:31 2023

@author: ayaha
"""
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from flatten_json import flatten_json

""" IMPORT DATA """
data = open('trainingdata\data.json')
loaded_data = json.load(data)
flat = loaded_data

num_items = 5
"""
[[{'altitude': 22, 'heading': 32, 'id': 'red', 'dataTrail': {}, 'type': 0, 'startPos': {'x': 226, 'y': 93}, 'intent': {'desiredHeading': 270, 'desiredAlt': 0, 'desiredSpeed': 450, 'desiredLoc': []}, 'capping': False}
,{'altitude': 29, 'heading': 32, 'id': 'red', 'dataTrail': {}, 'type': 0, 'startPos': {'x': 232.7843847692514, 'y': 97.23935411386564}, 'intent': {'desiredHeading': 270, 'desiredAlt': 0, 'desiredSpeed': 450, 'desiredLoc': []}, 'capping': False},
{'altitude': 32, 'heading': 32, 'id': 'red', 'dataTrail': {}, 'type': 0, 'startPos': {'x': 239.5687695385028, 'y': 101.47870822773129}, 'intent': {'desiredHeading': 270, 'desiredAlt': 0, 'desiredSpeed': 450, 'desiredLoc': []}, 'capping': False}]]
data is labeled as such
0-n pictures
each picture has 0-m groups
each group has heading, starting position(x,y), altitude
'n_pic' --> Label y
data x as follows:
['n_groups_i_j_startPos_x', 'n_groups_i_j_startPos_y',
    'n_groups_i_j_altitude', 'n_groups_i_j_heading']
where i is the number of groups or shapes in the picture

"""

"""Extract numerical information"""
Y = np.transpose(np.array([]))

# possibly the angle
found_xPos = False
found_yPos = False

startPositions = {}

n = 0
for key in flat:
    groups = key.get("groups")
    startPositions[n] = {}
    for i in range(0, len(groups)):
        newx = []
        newy = []
        # startPositions[n]["x"] = []
        # startPositions[n]["y"] = []
        for j in range(0, len(groups[i])):
            startPos = groups[i][j].get('startPos')
            newx.append(startPos.get("x"))
            newy.append(startPos.get("y"))
        startPositions[n]["x"] = newx
        startPositions[n]["y"] = newy
        # np.append(startPositions[n]["x"], startPos.get("x"))
        # np.append(startPositions[n]["y"], startPos.get("y"))
    n = n+1

    if 'pic' in key:
        val = key.get("pic")
        if 'AZIMUTH' in val:
            Y = np.append(Y, 'AZIMUTH')
        elif 'RANGE' in val:
            Y = np.append(Y, 'RANGE')
        elif 'WALL' in val:
            Y = np.append(Y, 'WALL')
        elif 'LADDER' in val:
            Y = np.append(Y, 'LADDER')
        elif 'CHAMPAGNE' in val:
            Y = np.append(Y, 'CHAMPAGNE')
        elif 'VIC' in val:
            Y = np.append(Y, 'VIC')
        elif 'SINGLE' in val:
            Y = np.append(Y, 'SINGLE')
"""
Labels
1: Azimuth, 2: Range, 3: Wall
4: Ladder, 5: Champagne 6: Vic
7: Single
"""

"""
Take X1 and X2 ---> matrix of points
"""
"""
I want one plot for every set of points in the same group {n}
"""
""" plot for ML """

# Create a dictionary to store X1 and X2 values for each n_group
# Create scatter plots for each n_group
file_path = os.path.join('output', 'Y.txt')
img_size = (500, 500)
save_dir = 'output'
np.savetxt(file_path, Y, fmt='%s')


for n in range(num_items):
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(startPositions[n].get(
        "x"), startPositions[n].get("y"), c='black', marker='3')
    ax.set_title("")  # (f"Group {n}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.set_xlabel("X1")
    # ax.set_ylabel("X2")
    # ax.set_rlim(0, 100)
    # ax.set_thetalim(0, 2*np.pi)
    ax.set_xlim(0, 400)
    ax.set_ylim(-200, 200)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    file_name = f"group_{n}.png"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)

    plt.close(fig)
    with Image.open(file_path) as img:
        center_x = img.width // 2
        center_y = img.height // 2

        left = max(0, center_x - img_size[0] // 2)
        right = min(img.width, center_x + img_size[0] // 2)
        top = max(0, center_y - img_size[1] // 2)
        bottom = min(img.height, center_y + img_size[1] // 2)

        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(file_path)
