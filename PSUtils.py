

import json

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
