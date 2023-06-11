# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:15:48 2023
@author: ayaha

Diplay histogram /stats analysis of underlying PS data
"""
import json
import logging
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from alive_progress import alive_bar

from PSLogger import psLog
from PSUtils import get_label, load_data

""" IMPORT DATA """
loaded_data = load_data()

Y = []

psLog.setLevel(logging.INFO)

# Compute and display the histogram
with alive_bar(len(loaded_data)) as bar:
    psLog.info("Generating histogram bins...")
    for k in loaded_data:
        # Make the answer key (bins)
        Y = np.append(Y, get_label(k))
        bar()

mpl.use("TkAgg")
hist = plt.hist(Y, 7)
plt.show()

# Compute and display any labels that occur more than the average
# (outside of 2 standard deviations)
deviations = scipy.stats.zscore(hist[0], axis=0)

outliers = []
n = 0
MAX_ALLOWED_DEVS = 2
for x in deviations:
    if (abs(x) > MAX_ALLOWED_DEVS):
        outliers.append(n)
    n += 1

if len(outliers) == 0:
    psLog.info(u'\u2713' + " Data within 2 standard deviations")
else:
    psLog.warning("X Data not within 2 standard deviations")
    psLog.warning("Outliers: ")
    for x in outliers:
        psLog.warning(labels[x])
