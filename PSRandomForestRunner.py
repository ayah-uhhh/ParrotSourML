# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha
"""
import PSRandomForest as psrf
import time
from tqdm import tqdm

debug = False

print("---------------")

_, time_elapsed, error = psrf.randomforest(
    img_size=19, debug=True)

print("------------------------------")
