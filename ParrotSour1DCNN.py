# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:30:25 2023

@author: ayaha
"""
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
# Load the data
pathtoimages = '\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000'
os.chdir(pathtoimages)
pictures = []
for n in range (0,1000):
    filename = f"group_{n}.png"
    image = Image.open(filename)
    image_array = np.array(image).flatten()
    pictures.append(image_array)
X = np.asarray(pictures)
Y = np.loadtxt('\\Users\\ayaha\\OneDrive\\Documents\\ParrotSour\\TrainingData\\1000\\Y.txt',dtype=str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Reshape the data to fit the input shape of the 1D-CNN model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))