import os
import pandas as pd
import numpy as np

# Define the custom function to calculate root sum square
def root_sum_square(row, start_col, end_col):
    selected_cols = row[start_col:end_col]
    rss = np.sqrt(np.sum(np.square(selected_cols)))
    return rss

# Define the path to the root folder
root_folder = "D:/toyota/いじるよう/sub5/fp"

# Recursively traverse all folders and subfolders
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(dirpath, filename)
            df = pd.read_csv(file_path, skiprows=range(1, 11))

            # Calculate root sum square of columns 2 to 4 and 5 to 8
            rss_2_4 = df.apply(root_sum_square, axis=1, start_col=2, end_col=5)
            rss_5_8 = df.apply(root_sum_square, axis=1, start_col=5, end_col=9)

            # Add the calculated values to columns 16 and 17
            df.loc[:, 16] = rss_2_4
            df.loc[:, 17] = rss_5_8

            # Write the modified DataFrame to a new CSV file
            new_file_path = os.path.join(dirpath, 'modified_' + filename)
            df.to_csv(new_file_path, index=False)

            # Alternatively, you can overwrite the original file by uncommenting the following line:
            # df.to_csv(file_path, index=False)

