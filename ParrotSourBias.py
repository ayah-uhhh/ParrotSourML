# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:15:48 2023

@author: ayaha
"""
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import json
from flatten_json import flatten_json

datab = open('\\Users\\ayaha\\OneDrive\\Documents\\MachineLearning\\data1000.json')
data1000=json.load(datab)
flat = flatten_json(data1000)
Y = np.transpose(np.array([]))


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
            print(Y)
            break 
hist(Y,7)