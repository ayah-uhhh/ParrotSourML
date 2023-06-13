# -*- coding: utf-8 -*-
"""
Created on 10 Jun 2023
@author: ayaha

This file runs a single RandomForest prediction with the coded img_size parameter.
Intended to be run with the best img_size found via the Pooled RF.
"""
# import joblib
import pickle
import os
import json
import logging

import ParrotSourSVM as psvm
from PSLogger import psLog

psLog.setLevel(logging.DEBUG)

psLog.info("------------------------------")

_, time_elapsed, error = psvm.psSVM("rbf", 15, "ovr", show_cm=True)

with open('PSSVMSaved.pkl', 'wb') as file:
    pickle.dump(psvm.psSVM, file)

psLog.info("------------------------------")


# joblib.dump(psvm.psSVM, 'PSSVMSaved.pkl')

# os.path.exists('PSSVMSaved.pkl')
