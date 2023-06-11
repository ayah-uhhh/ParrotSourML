"""tf trial"""
import tensorflow as tf
import os
import json
import csv

"""Import Data"""
with open("trainingdata\start_positions.json", "r") as file:
    startPositions = json.load(file)

ds_file = tf.data.TextLineDataset("trainingdata\start_positions.json")
