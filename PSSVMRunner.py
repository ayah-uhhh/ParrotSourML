# -*- coding: utf-8 -*-
"""
Created on 10 Jun 2023
@author: ayaha

This file runs a single RandomForest prediction with the coded img_size parameter.
Intended to be run with the best img_size found via the Pooled RF.
"""
import joblib
import pickle
import logging

import ParrotSourSVM as psvm
from PSLogger import psLog

psLog.setLevel(logging.DEBUG)

psLog.info("------------------------------")

_, time_elapsed, error = psvm.psSVM("rbf", 15, "ovr", show_cm=True)

psLog.info("------------------------------")

with open('PSSVMSaved.pkl', 'wb') as file:
    pickle.dump(psvm.psSVM, file)
# joblib.dump(psvm.psSVM, 'PSSVMSaved.pkl')
