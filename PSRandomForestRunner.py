# -*- coding: utf-8 -*-
"""
Created on 10 Jun 2023
@author: ayaha

This file runs a single RandomForest prediction with the coded img_size parameter.
Intended to be run with the best img_size found via the Pooled RF.
"""
import time

from tqdm import tqdm

import PSRandomForest as psrf

debug = False

print("------------------------------")

_, time_elapsed, error = psrf.randomforest(
    img_size=15, debug=True)

print("------------------------------")
