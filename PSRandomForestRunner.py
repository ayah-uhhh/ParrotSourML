# -*- coding: utf-8 -*-
"""
Created on 10 Jun 2023
@author: ayaha

This file runs a single RandomForest prediction with the coded img_size parameter.
Intended to be run with the best img_size found via the Pooled RF.
"""
import logging

import PSRandomForest as psrf
from PSLogger import psLog

psLog.setLevel(logging.DEBUG)

psLog.info("------------------------------")

_, time_elapsed, error, _ = psrf.randomforest(
    img_size=15, show_cm=True, save=True)

psLog.info("------------------------------")
