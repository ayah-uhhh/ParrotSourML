# -*- coding: utf-8 -*-

"""

Created on Sun Apr 23 08:46:39 2023


@author: ayaha

"""

from sklearn.datasets import make_classification

import numpy as np

import matplotlib.pyplot as plt
import os

from PIL import Image

from sklearn.model_selection import train_test_split

from sklearn import svm, metrics

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
import time

from alive_progress import alive_bar


def randomforest(img_size=30, n_estimators=240, use_pca=False, debug=False):

    start_time = time.time()

    pictures = []

    for root, dirs, files in os.walk("./output", topdown=True):

        filelist = []

        for n in files:

            if ("Y.txt" not in n):

                filelist.append(os.path.splitext(n)[0])

        sorted_files = sorted(filelist, key=int)

        for name in sorted_files:

            if ("Y.txt" not in name):

                image = Image.open(os.path.join(root, name)+'.png')

                resized_image = image.resize((img_size, img_size))

                image_array = np.array(resized_image).flatten()

                pictures.append(image_array)

    baseestimator = RandomForestClassifier(n_estimators=n_estimators)

    Y = np.loadtxt('output\\Y.txt', dtype=str)

    X = np.asarray(pictures)

    x_pca = None

    num_components = len(pictures) if len(pictures) < 7 else 7

    if (use_pca):

        pca = PCA(n_components=num_components)

        x_pca = pca.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca if use_pca else X, Y, test_size=0.2, shuffle=False)
    baseestimator.fit(x_train, y_train)
    predicted = baseestimator.predict(x_test)

    elapsed_time = time.time() - start_time

    error = 1 - metrics.accuracy_score(y_test, predicted)

    if (debug):

        print("Time taken to classify:", elapsed_time, "seconds")
        print(f"Classification error: {error}")
        disp = metrics.ConfusionMatrixDisplay.from_predictions(

            y_test, predicted)

        disp.figure_.suptitle("Confusion Matrix")

        plt.show()

    return [img_size, elapsed_time, error]
