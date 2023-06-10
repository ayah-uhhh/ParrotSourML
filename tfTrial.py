"""tf trial"""
import tensorflow as tf
import os
import json
import csv

"""Import Data"""
with open("trainingdata\start_positions.json", "r") as file:
    startPositions = json.load(file)

# Define the CSV file path
csv_file = "start_positions.csv"

# Open the CSV file and write the data
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["key", "x", "y"])

    # Write the data rows
    for key, values in startPositions.items():
        x_values = values.get("x", [])
        y_values = values.get("y", [])
        writer.writerow(
            [
                key,
                ",".join(str(x) for x in x_values),
                ",".join(str(y) for y in y_values),
            ]
        )
