# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:47:31 2023

@author: ayaha
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from flatten_json import flatten_json

""" IMPORT DATA """
data = open('\\Users\\ayaha\\OneDrive\\Documents\\MachineLearning\\data1000.json')
data1000=json.load(data)
#print(data5)
#plot(data5)

""" DATA MANIPULATION ATTEMPT 1 """
# data1 = pd.json_normalize(data5)
# pd.json_normalize(data5,max_level=0)
# ohboy = pd.json_normalize(data5,record_path=['startPos'],meta =['x','y'])

""" DATA MANIPULATION ATTEMPT 2 """
"""flatten"""
flat = flatten_json(data1000)
#print(flat) 
    

""" DATA MANIPULATION ATTEMPT 3 """
"""convert to a dataframe"""
# df = pd.DataFrame.from_dict(data5)
# print(df)
# df[['altitude','heading','id','dataTrail','type','startPos','intent']]=df["groups"].apply(lambda x: pd.Series(str(x).split(",")))
# df2 = pd.read_json(df["groups"],orient='columns',typ='series')
# print(df2)


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
 ['n_groups_i_j_startPos_x', 'n_groups_i_j_startPos_y', 'n_groups_i_j_altitude', 'n_groups_i_j_heading']
 where i is the number of groups or shapes in the picture
 
 """
"""extract relevant numerical information: INTUITION """
#y = array[]
#x = array[]
# for n in range (0,4):
#     y = (df(n))
#     for i in range (0,10):
#         for j in range(0,10):
#             x = (fdata5(n_groups_i_j_startPos_x, n_groups_i_j_startPos_y, n_groups_i_j_altitude, n_groups_i_j_heading))
"""practicing extraction"""

# for key in flat:
#     # if '_heading' in key: # I also think we dont need heading right now
#     #     print(heading)
   
#     if 'startPos_x' in key:
#         X1 = np.append(X1, flat[key])
        
#     if 'startPos_y' in key:
#         X2 = np.append(X2, flat[key])
       
#     # if '_altitude' in key:        # I do not think altitude will be needed initially
#     #     altitude = (key, flat[key])
#     #     #print(altitude)      
#     if '_pic' in key:
#         y = (key, flat[key])
#         #print(y)
"""Extract numerical information"""
X1 = np.transpose(np.array([]))
X2 = np.transpose(np.array([]))
# #X3 = np.transpose(np.array([]))
Y = np.transpose(np.array([]))
found_xPos = False
for key in flat:
    if '_startPos_x' in key:
       found_xPos = True
       for n in range (0,1000):
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
        for n in range (0,1000):
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

# found_altitude = False
# for key in flat:
#     if '_altitude' in key:
#         found_altitude = True
#         for n in range (0,4):
#             for i in range(0,5):
#                 for j in range(0,5):
#                     key = f"{n}_groups_{i}_{j}_altitude"
#                     try:
#                         X3 = np.append(X3, flat[key])
#                     except KeyError:
#                         continue
#         if len(X3) == 0:
#             break
#         else:
#             print(X3)
#             break
"""
 Labels
 1: Azimuth, 2: Range, 3: Wall
 4: Ladder,  5: Champagne 6: Vic
 7: Single
 """
found_label = False
for key in flat:
    if '_pic' in key:
        found_label = True
        for n in range (0,1000):
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
# found_label = False
# for key in flat:
#     if '_pic' in key:
#         found_label = True
#         for n in range (0,1000):
#             key = f"{n}_pic"
#             try:
#                 val = flat[key]
#                 if 'AZIMUTH' in val:
#                     Y = np.append(Y,1)
#                 elif 'RANGE' in val:
#                     Y = np.append(Y,2)
#                 elif 'WALL' in val:
#                     Y = np.append(Y,3)
#                 elif 'LADDER' in val:
#                     Y = np.append(Y,4)
#                 elif 'CHAMPAGNE' in val:
#                     Y = np.append(Y,5)
#                 elif 'VIC' in val:
#                     Y = np.append(Y,6)
#                 elif 'SINGLE' in val:
#                     Y = np.append(Y,7)
#             except KeyError:
#                 continue
#         if len(Y) == 0:
#             break
#         else:
#             #print(Y)
#             break 
# found_label = False
# for key in flat:
#     if '_pic' in key:
#         found_label = True
#         for n in range (0,1000):
#             key = f"{n}_pic"
#             try:
#                 Y = np.append(Y, flat[key])
#             except KeyError:
#                 continue
#         if len(Y) == 0:
#             break
#         else:
#             #print(Y)
#             break 


""" 
Take X1 and X2 ---> matrix of points
"""
"""
I want one plot for every set of points in the same group {n}
"""
""" plot for ML """

# Create a dictionary to store X1 and X2 values for each n_group
X1_dict = {}
X2_dict = {}
#possibly the angle
found_xPos = False
for key in flat:
    if '_startPos_x' in key:
        found_xPos = True
        for n in range (0,1000):
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
        for n in range (0,1000):
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
file_path = os.path.join('\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000','Y.txt')
from PIL import Image
img_size = (500,500)
save_dir = '\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000'
np.savetxt(file_path, Y, fmt='%s')
for n in range(1000):
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



