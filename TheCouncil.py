# load models and give them weights
import logging
import multiprocessing as mp
import os
import time

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from ParrotSourPreProcessor import preprocess
from PSLogger import psLog

"""        IMAGE         """
import(IMAGE)
preprocess()

"""        SVM          """
loaded_model, loaded_model_settings = joblib.load(
    open('PSSVMSaved.jbl', 'rb'))
SVM = loaded_model(probability = True)

confidence_scores = clf.decision_function(IMAGE)
print(confidence_scores)

"""        Random Forest          """
# loaded_model2, img_size = joblib.load(open('PSRandomForestSaved.jbl', 'rb'))
# forest = loaded_model2
