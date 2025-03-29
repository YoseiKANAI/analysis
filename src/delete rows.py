import os
import csv

folder_path = "C:/Users/ShimaLab/Desktop/いじるよう"

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                del rows[:26]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
