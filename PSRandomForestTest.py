# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023
@author: ayaha

Defines a function to run a single RandomForest prediction with the coded img_size parameter.
"""
import logging
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from PSUtils import IMAGE_DIR, OUT_DIR, get_pics
from PSLogger import psLog

start_time = time.time()
pictures = []
img_size = 15
# load the answer key and format the dataset
Y = np.loadtxt(OUT_DIR+"\\Y.txt", dtype=str)
X = get_pics(img_size)

"""
LOAD MODEL
"""
loaded_model = joblib.load(open('PSRandomForestSaved.jbl', 'rb'))
forest = loaded_model

# predict based on test data
predicted = forest.predict(X)

# Results:
elapsed_time = time.time() - start_time

error_rate = 1 - metrics.accuracy_score(Y, predicted)

psLog.debug("Time taken to classify: %s seconds", elapsed_time)
psLog.debug(f"Classification error: {error_rate}")


disp = metrics.ConfusionMatrixDisplay.from_predictions(
    Y, predicted)
disp.figure_.suptitle("Confusion Matrix")
plt.show()
