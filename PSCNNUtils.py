import os

import numpy as np
from PIL import Image

from PSLogger import psLog
from PSUtils import IMAGE_DIR


def one_hot_encode_labels(answerkey=os.path.join('output', 'Y.txt')):
    Y = np.loadtxt(answerkey, dtype=str)

    # one hot encode
    labelmap = {'AZIMUTH': 0, 'RANGE': 1, 'WALL': 2,
                'LADDER': 3, 'CHAMPAGNE': 4, 'VIC': 5, 'SINGLE': 6}
    return np.array(list(map(labelmap.get, Y)))


def get_cnn_pics(img_dir=IMAGE_DIR):
    pictures = []

    psLog.debug("Loading CNN images...")
    # read the output image directory to prep the dataset
    filelist = []
    for root, _, files in os.walk(img_dir, topdown=True):
        for n in files:
            filelist.append(os.path.splitext(n)[0])
    sorted_files = sorted(filelist, key=int)

    # read the images for form the dataset
    for name in sorted_files:
        image = Image.open(os.path.join(root, name)+'.png')
        resized_image = image.resize((100, 100))
        pictures.append(np.array(resized_image))
    return np.asarray(pictures)
