# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 08:14:25 2023

@author: ayaha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from flatten_json import flatten_json

""" IMPORT DATA """
data = open('\\Users\\ayaha\\OneDrive\\Documents\\MachineLearning\\data1.json')
data1000=json.load(data)
flat = flatten_json(data1000)
"""Extract numerical information"""
X1 = np.transpose(np.array([]))
X2 = np.transpose(np.array([]))
# #X3 = np.transpose(np.array([]))
Y = np.transpose(np.array([]))
found_xPos = False
for key in flat:
    if '_startPos_x' in key:
       found_xPos = True
       for n in range (0,5):
            for i in range(0,10):
                for j in range(0,10):
                    key = f"{n}_groups_{i}_{j}_startPos_x" #n is picture number, i is cluster of air craft, j is the individual aircraft
                    try:
                        X1 = np.append(X1, flat[key])
                    except KeyError:
                        continue
       if len(X1) == 0:
            break
       else:
            break

found_yPos = False
for key in flat:
    if '_startPos_y' in key:
        found_yPos = True
        for n in range (0,5):
            for i in range(0,10):
                for j in range(0,10):
                    key = f"{n}_groups_{i}_{j}_startPos_y"#n is picture number, i is cluster of air craft, j is the individual aircraft
                    try:
                        X2 = np.append(X2, flat[key])
                    except KeyError:
                        continue
        if len(X2) == 0:
            break
        else:
            #print(X2)
            break
found_label = False
for key in flat:
    if '_pic' in key:
        found_label = True
        for n in range (0,5):
            key = f"{n}_pic"
            try:
                val = flat[key]
                if 'AZIMUTH' in val:
                    Y = np.append(Y,'AZIMUTH')
                elif 'RANGE' in val:
                    Y = np.append(Y,'RANGE')
                elif 'WALL' in val:
                    Y = np.append(Y,'WALL')
                elif 'LADDER' in val:
                    Y = np.append(Y,'LADDER')
                elif 'CHAMPAGNE' in val:
                    Y = np.append(Y,'CHAMPAGNE')
                elif 'VIC' in val:
                    Y = np.append(Y,'VIC')
                elif 'SINGLE' in val:
                    Y = np.append(Y,'SINGLE')
            except KeyError:
                continue
        if len(Y) == 0:
            break
        else:
            #print(Y)
            break 
# Create a dictionary to store X1 and X2 values for each n_group
X1_dict = {}
X2_dict = {}
#possibly the angle
found_xPos = False
for key in flat:
    if '_startPos_x' in key:
        found_xPos = True
        for n in range (0,5):
            X1_dict[n] = []
            for i in range(0,10):
                for j in range(0,10):
                    key = f"{n}_groups_{i}_{j}_startPos_x" #n is picture number, i is cluster of air craft, j is the individual aircraft
                    try:
                        X1_dict[n].append(flat[key])
                    except KeyError:
                        continue
        if len(X1_dict[0]) == 0:
            break
        else:
            break
#radius maybeius
found_yPos = False
for key in flat:
    if '_startPos_y' in key:
        found_yPos = True
        for n in range (0,5):
            X2_dict[n] = []
            for i in range(0,10):
                for j in range(0,10):
                    key = f"{n}_groups_{i}_{j}_startPos_y"#n is picture number, i is cluster of air craft, j is the individual aircraft
                    try:
                        X2_dict[n].append(flat[key])
                    except KeyError:
                        continue
        if len(X2_dict[0]) == 0:
            break
        else:
            break

# Create scatter plots for each n_group
import os 
from PIL import Image
img_size = (500,500)
file_path = os.path.join('\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\5','Y.txt')

save_dir = '\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\5'
np.savetxt(file_path, Y, fmt='%s')
for n in range(0,5):
    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(X1_dict[n], X2_dict[n],c='black', marker='3')
    ax.set_title("")#(f"Group {n}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    #ax.set_xlabel("X1")
    #ax.set_ylabel("X2")
    #ax.set_rlim(0, 100)
    #ax.set_thetalim(0, 2*np.pi)
    ax.set_xlim(0,400)
    ax.set_ylim(-200,200)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    file_name = f"group_{n}.png"
    file_path = os.path.join(save_dir,file_name)
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
        