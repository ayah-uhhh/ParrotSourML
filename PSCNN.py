import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from PSCNNUtils import get_cnn_pics, one_hot_encode_labels
from PSLogger import psLog


psLog.setLevel(logging.DEBUG)


def pscnn(optimizer='rmsprop', filters=3, kernel_size=(3, 3), img_size=100, show_chart=False, save=False, epochs=150, batch_size=32, probability=True):
    # optimizer = 'nadam', 'rmsprop', 'adam'
    """Import Data"""

    psLog.debug("Preprocessing training data...")
    start_time = time.time()
    X = get_cnn_pics()
    Y = one_hot_encode_labels()
    psLog.debug("Created training dataset. (%.2f)", time.time()-start_time)

    psLog.debug("Splitting data...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)

    psLog.debug("Building model...")
    start_time = time.time()
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

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=optimizer, metrics=['accuracy'])

    psLog.debug("Built model. (%.2f)", time.time()-start_time)

    model.summary()
    # Train the model
    psLog.debug("Training model...")
    history = model.fit(X_train, y_train, epochs, batch_size,
                        validation_data=(X_test, y_test))
    psLog.debug("Model trained. (%.2f)", time.time()-start_time)

    elapsed_time = time.time()-start_time

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

    if save:
        psLog.debug("Saving model...")
        start_time = time.time()
        model.save('ps_cnn_model.h5')
        psLog.debug("Saved model (%.2fs)", time.time()-start_time)

    psLog.debug('Accuracy: %.2f', (accuracy*100))
    psLog.debug("Loss: %s", loss)

    if probability:
        predictions = model.predict(X_test)
        confidence = np.max(predictions, axis=1) * 100
        labels = np.argmax(predictions, axis=1)
        for conf, label in zip(confidence, labels):
            print(f"CNN Confidence: {conf:.2f}%, Label: {label}")

    return [(optimizer, filters, kernel_size, img_size), elapsed_time, loss, accuracy, model]


pscnn()
