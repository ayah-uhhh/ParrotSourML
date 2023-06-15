

import json
import os

import numpy as np
from PIL import Image

OUT_DIR = "output"
IMAGE_DIR = os.path.join(OUT_DIR, "images")
ANSWER_FILE = os.path.join(OUT_DIR, "Y.txt")
LABELS = ['AZIMUTH', 'RANGE', 'WALL',
          "LADDER", "CHAMPAGNE", "VIC", "SINGLE"]


def load_data(filename="data1000.json"):
    data = open(os.path.join("trainingdata", filename))
    return json.load(data)


def get_label(k):
    if "pic" in k:
        val = k.get("pic")
        found_labels = [label for label in LABELS if (label in val)]
        return found_labels[0] if len(
            found_labels) > 0 else "UNKNOWN"


def get_pics(img_size=15, imgdir=IMAGE_DIR, slice_len=-1):
    pictures = []

    # read the output image directory to prep the dataset
    filelist = []
    for _, _, files in os.walk(imgdir, topdown=True):

        for n in files:
            filelist.append(os.path.splitext(n)[0])
    sorted_files = sorted(filelist, key=int)

    if (slice_len != -1):
        print("TODO - Pick out random sample of files")

    # read the images for form the dataset
    for name in sorted_files:
        image = Image.open(os.path.join(imgdir, name)+'.png')
        resized_image = image.resize((img_size, img_size))
        image_array = np.array(resized_image).flatten()
        pictures.append(image_array)
    return np.asarray(pictures)
