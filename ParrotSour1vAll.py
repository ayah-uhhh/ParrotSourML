# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023

@author: ayaha
"""
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

start_time = time.time()

pictures = []
for root, dirs, files in os.walk("./output", topdown=False):
    for name in files:
        if ("Y.txt" not in name):
            image = Image.open(os.path.join(root, name))
            resized_image = image.resize((100, 100))
            image_array = np.array(resized_image).flatten()
            pictures.append(image_array)

baseestimator = RandomForestClassifier(n_estimators=100, max_depth=10)


Y = np.loadtxt('output\\Y.txt', dtype=str)
X = np.asarray(pictures)

num_components = len(pictures) if len(pictures) < 7 else 7
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, Y, test_size=0.2, shuffle=False)
baseestimator.fit(X_train, y_train)
predicted = baseestimator.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time

print("Time taken to classify:", elapsed_time, "seconds")

error = 1 - metrics.accuracy_score(y_test, predicted)
print(f"Classification error: {error}")


debug = True
if (debug):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
