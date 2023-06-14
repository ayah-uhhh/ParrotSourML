"""tf trial"""
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split

from PSLogger import psLog
from PSUtils import IMAGE_DIR, OUT_DIR

psLog.setLevel(logging.DEBUG)


def pscnn(optimizer='rmsprop', filters=3, kernel_size=(3, 3), img_size=100, show_chart=False):
    # optimizer = 'nadam', 'rmsprop', 'adam'
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

    Y = np.loadtxt(os.path.join(OUT_DIR, "Y.txt"), dtype=str)

    # one hot encode
    from keras.utils import to_categorical
    labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2,
                'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}
    Y = np.array(list(map(labelmap.get, Y)))

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
    model.add(Conv2D(filters, kernel_size,
              input_shape=(img_size, img_size, 4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters, kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters, kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters, kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters, kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Fully Connected Layer: This layer takes the lessons learned from the Convolutional Layer
    # and the smaller feature map form the Pooling Layer and combines them in order to make a prediction.
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    psLog.debug("Compiling model...")
    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    # Train the model
    psLog.debug("Training model...")
    history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                        validation_data=(X_test, y_test))

    if (show_chart):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    psLog.debug("Saving model...")
    start_time = time.time()
    model.save('ps_cnn_model.h5')
    psLog.debug("Saved model (%.2fs)", time.time()-start_time)

    psLog.debug('Accuracy: %.2f', (accuracy*100))
    psLog.debug("Loss: %s", loss)


pscnn()
