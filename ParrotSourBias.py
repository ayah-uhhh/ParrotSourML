# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:15:48 2023

@author: ayaha
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
from alive_progress import alive_bar
import json
import sys
import re

""" IMPORT DATA """
data = open("trainingdata\data1000.json")
loaded_data = json.load(data)

Y = []


def zscore(s):
    return (s - np.mean(s, axis=0)) / np.std(s, axis=0)


labels = ['AZIMUTH', 'RANGE', 'WALL', "LADDER", "CHAMPAGNE", "VIC", "SINGLE"]

"""Extract label information"""
with alive_bar(len(loaded_data)) as bar:
    print("Generating histogram bins...")
    for k in loaded_data:  # pictures

        # Make the answer key
        if "pic" in k:
            val = k.get("pic")

            found_label = [label for label in labels if (label in val)][0]
            Y = np.append(Y, found_label)

        bar()

mpl.use("TkAgg")
hist = plt.hist(Y, 7)

deviations = scipy.stats.zscore(hist[0], axis=0)

outliers = []
n = 0
MAX_ALLOWED_DEVS = 2
for x in deviations:
    if (abs(x) > MAX_ALLOWED_DEVS):
        outliers.append(n)
    n += 1

if len(outliers) == 0:
    print(u'\u2713' + " Data within 2 standard deviations")
else:
    print("X Data not within 2 standard deviations")
    print("Outliers: ")
    for x in outliers:
        print(labels[x])
plt.show()
