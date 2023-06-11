"""tf trial"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split

from PSLogger import psLog
from PSUtils import IMAGE_DIR, OUT_DIR

psLog.setLevel(logging.DEBUG)


def PSCNN(filters=64, kernel_size=(4, 4), img_size=100):
    """Import Data"""

    pictures = []

    # read the output image directory to prep the dataset
    filelist = []
    for root, dirs, files in os.walk(IMAGE_DIR, topdown=True):

        for n in files:
            filelist.append(os.path.splitext(n)[0])
    sorted_files = sorted(filelist, key=int)

    psLog.debug("Loading images...")
    # read the images for form the dataset
    for name in sorted_files:
        image = Image.open(os.path.join(root, name)+'.png')
        resized_image = image.resize((img_size, img_size))
        pictures.append(np.array(resized_image))
    X = np.asarray(pictures)

    psLog.debug("Converting to binary...")
    # X = X / 255.0

    # plt.figure(figsize=(img_size, img_size))
    # for i in range(5):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(X[i])
    #     # The CIFAR labels happen to be arrays,
    #     # which is why you need the extra index
    #     # plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()

    Y = np.loadtxt(OUT_DIR+"\\Y.txt", dtype=str)

    """
    one hot encode
    """
    from keras.utils import to_categorical
    labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2,
                'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}
    Y = np.array(list(map(labelmap.get, Y)))
    # Y = to_categorical(Y, num_classes=7)
    """
    //one hot encode
    """

    psLog.debug("Splitting data...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)

    psLog.debug("Building model...")
    # Define the model
    model = Sequential()

    # Convolutional Layer: This layer creates a feature map by applying filters to the input image
    # and computing the dot product between the filtered weights and the pixel values.
    # The feature map shows what pixels are the most important when classifying an image.
    # Pooling: This layer reduces the size of the feature map  by averaging pixels that are near each other.
    model.add(Conv2D(3, (3, 3),
              input_shape=(img_size, img_size, 4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(3, (3, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(3, (3, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(3, (3, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(3, (3, 3)))

    # Fully Connected Layer: This layer takes the lessons learned from the Convolutional Layer
    # and the smaller feature map form the Pooling Layer and combines them in order to make a prediction.
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation="softmax"))

    psLog.debug("Compiling model...")
    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    # Train the model
    psLog.debug("Training model...")
    model.fit(X_train, y_train, epochs=100, batch_size=45,
              validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))


PSCNN()
