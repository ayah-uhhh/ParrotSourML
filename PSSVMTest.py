# -*- coding: utf-8 -*-
"""
Created on 10 Jun 2023
@author: ayaha

This file runs a single RandomForest prediction with the coded img_size parameter.
Intended to be run with the best img_size found via the Pooled RF.
"""
import joblib
import pickle
import numpy as np
import ParrotSourSVM as psvm
from PSLogger import psLog
from PSUtils import IMAGE_DIR, OUT_DIR, get_pics
import time
from sklearn import metrics, svm
import matplotlib.pyplot as plt


start_time = time.time()

Y = np.loadtxt(OUT_DIR+'\\Y.txt', dtype=str)
X = get_pics(100)  # manual image size

"""
LOAD MODEL
"""
loaded_model, loaded_model_settings = joblib.load(open('PSSVMSaved.jbl', 'rb'))
clf = loaded_model
kernel, sea, shape, size_img = loaded_model_settings
print("kernel =", kernel, "Sea =", sea,
      "shape =", shape, "size image =", size_img)

predicted = clf.predict(X)

error_rate = 1 - metrics.accuracy_score(Y, predicted)
end_time = time.time()
elapsed_time = end_time - start_time

disp = metrics.ConfusionMatrixDisplay.from_predictions(
    Y, predicted)
disp.figure_.suptitle("Confusion Matrix")
plt.show()
print("elasped time = ", elapsed_time, "error rate =", error_rate)
