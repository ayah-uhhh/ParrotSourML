# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:46:39 2023

@author: ayaha
"""
import numpy as np 
import matplotlib.pyplot as plt
import os
#import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import time

start_time = time.time()
pathtoimages = '\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000'
os.chdir(pathtoimages)

pictures = []
for n in range (0,1000):
    filename = f"group_{n}.png"
    image = Image.open(filename)
    resized_image = image.resize((100,100))
    image_array = np.array(resized_image).flatten()
    pictures.append(image_array)
baseestimator = RandomForestClassifier(n_estimators=100,max_depth=10) # svm.SVC(kernel="linear",C=1)#DecisionTreeClassifier(criterion='entropy',max_depth=5)

#from sklearn.ensemble import AdaBoostClassifier
#boosttree=AdaBoostClassifier(base_estimator=baseestimator,n_estimators=100)
Y = np.loadtxt('\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000\\Y.txt',dtype=str)
X = np.asarray(pictures)
#Print the loaded array
pca =PCA(n_components=7)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, shuffle=False)
baseestimator.fit(X_train, y_train)
predicted = baseestimator.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time

print("Time taken to classify:", elapsed_time, "seconds")

error = 1 - metrics.accuracy_score(y_test, predicted)
print(f"Classification error: {error}")

# Visualize the decision tree
#plt.figure(figsize=(20,20),dpi=200)
#plot_tree(baseestimator, class_names=['AZIMUTH', 'RANGE','LADDER','WALL','CHAMPAGNE','VIC','SINGLE'], filled=True)


