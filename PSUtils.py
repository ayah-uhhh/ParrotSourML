

import json
import os
from PIL import Image
import numpy as np

OUT_DIR = "output"
IMAGE_DIR = OUT_DIR+"\\images"
ANSWER_FILE = OUT_DIR+"\\Y.txt"
LABELS = ['AZIMUTH', 'RANGE', 'WALL',
          "LADDER", "CHAMPAGNE", "VIC", "SINGLE"]


def load_data(filename="data1000.json"):
    data = open("trainingdata\\"+filename)
    return json.load(data)


def get_label(k):
    if "pic" in k:
        val = k.get("pic")
        found_labels = [label for label in LABELS if (label in val)]
        return found_labels[0] if len(
            found_labels) > 0 else "UNKNOWN"


def get_pics(img_size=15):
    pictures = []

    # read the output image directory to prep the dataset
    filelist = []
    for root, dirs, files in os.walk(IMAGE_DIR, topdown=True):

        for n in files:
            filelist.append(os.path.splitext(n)[0])
    sorted_files = sorted(filelist, key=int)

    # read the images for form the dataset
    for name in sorted_files:
        image = Image.open(os.path.join(root, name)+'.png')
        resized_image = image.resize((img_size, img_size))
        image_array = np.array(resized_image).flatten()
        pictures.append(image_array)
    return np.asarray(pictures)
