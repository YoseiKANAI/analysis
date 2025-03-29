import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import global_value as g

def main():
    output_dir = output_preparation()

    ID = 10
    file_list = preparation(ID)
    for csv_file in file_list:
        df = pd.read_csv(csv_file)
        df["ax"] = lowpass_filter(df["ax"], sampling_rate=1000, cutoff_freq=50)
        df["ay"] = lowpass_filter(df["ay"], sampling_rate=1000, cutoff_freq=50)
        filename = os.path.basename(csv_file).replace(".csv", "")
        plot_cop(df, output_dir, ID, filename)

def preparation(ID):
    root_dir = f"D:/User/kanai/Data/{g.datafile}/result_COP/dump/sub{ID+1:02d}*.csv"
    file_list = glob.glob(root_dir, recursive=True)
    return file_list

def output_preparation():
    output_dir = f"D:/User/kanai/Data/{g.datafile}/result_COP/dump/plot_1axis_filtered"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def lowpass_filter(data, sampling_rate, cutoff_freq):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def plot_cop(df, output_dir, ID, filename):
    time = df.index / 1000  # assuming the sampling rate is 1000 Hz
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot x-axis vs time
    axs[0].plot(time, df["ax"], label="X-axis")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("X-axis displacement")
    # Plot y-axis vs time
    axs[1].plot(time, df["ay"], label="Y-axis")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Y-axis displacement")

    plt.tight_layout()
    out_name = f"{output_dir}/sub{ID+1}_{filename}.svg"
    plt.savefig(out_name)
    plt.close()

if __name__ == "__main__":
    main()
