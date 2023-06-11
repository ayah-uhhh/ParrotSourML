"""tf trial"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Dense, Conv2D
from sklearn.model_selection import train_test_split
import numpy as np
from PSUtils import OUT_DIR, IMAGE_DIR
from PIL import Image
import os
import skimage


def PSCNN(filters=64, kernel_size=(4, 4), img_size=15):
    """Import Data"""

    pictures = []

    # read the output image directory to prep the dataset
    filelist = []
    for root, dirs, files in os.walk(IMAGE_DIR, topdown=True):

        for n in files:
            filelist.append(os.path.splitext(n)[0])
    sorted_files = sorted(filelist, key=int)

    # read the images for form the dataset
    for name in sorted_files:
        im = skimage.io.imread(os.path.join(root, name)+'.png')
        # image = Image.open(os.path.join(root, name)+'.png')
        # resized_image = image.resize((img_size, img_size))
        # image_array = np.array(image)  # .flatten()
        pictures.append(im)
    X = np.asarray(pictures)
    X = X / 255.0

    # one hot encode
    Y = np.loadtxt(OUT_DIR+"\\Y.txt", dtype=str)
    from keras.utils import to_categorical
    labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2,
                'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}
    Yint = np.array(list(map(labelmap.get, Y)))
    Y = to_categorical(Yint, num_classes=7)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False)

    # Reshape the data to fit the input shape of the 1D-CNN model
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the model
    model = Sequential()
    model.add(Conv2D(filters, (3, 3), activation='relu',
              input_shape=(480, 640)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Convolutional Layer: This layer creates a feature map by applying filters to the input image
    # and computing the dot product between the filtered weights and the pixel values.
    # The feature map shows what pixels are the most important when classifying an image.
    # Pooling: This layer reduces the size of the feature map  by averaging pixels that are near each other.

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7))
    # Fully Connected Layer: This layer takes the lessons learned from the Convolutional Layer
    # and the smaller feature map form the Pooling Layer and combines them in order to make a prediction.

    # Compile the model
    model.compile(loss='mae',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))


PSCNN()
