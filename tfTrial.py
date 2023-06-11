"""tf trial"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from PSUtils import OUT_DIR, get_pics


def PSCNN(filters=64, kernel_size=5, img_size=15):
    """Import Data"""

    X = get_pics(img_size)

    # one hot encode
    Y = np.loadtxt(OUT_DIR+"\\Y.txt", dtype=str)
    # from keras.utils import to_categorical
    # labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2, 'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}
    # Yint = np.array(list(map(labelmap.get,Y)))
    # Y = to_categorical(Yint,num_classes=7)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)

    # Reshape the data to fit the input shape of the 1D-CNN model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the model
    model = Sequential()
    model.add(Conv1D(filters, kernel_size, activation='relu',
              input_shape=(X_train.shape[1], 1)))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    # Convolutional Layer: This layer creates a feature map by applying filters to the input image
    # and computing the dot product between the filtered weights and the pixel values.
    # The feature map shows what pixels are the most important when classifying an image.

    model.add(MaxPooling1D(pool_size=3))
    # Pooling: This layer reduces the size of the feature map  by averaging pixels that are near each other.

    model.add(Flatten())
    model.add(Dense(7, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # Fully Connected Layer: This layer takes the lessons learned from the Convolutional Layer
    # and the smaller feature map form the Pooling Layer and combines them in order to make a prediction.

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))


PSCNN()
