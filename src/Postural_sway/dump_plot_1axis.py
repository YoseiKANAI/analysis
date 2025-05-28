import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import global_value as g

def main():
    output_dir_filtered = output_preparation("plot_1axis_filtered")
    output_dir_raw = output_preparation("plot_1axis_raw")

    ID = 10
    file_list = preparation(ID)
    for csv_file in file_list:
        df = pd.read_csv(csv_file)
        # --- 生波形プロット ---
        filename = os.path.basename(csv_file).replace(".csv", "")
        #plot_cop(df, output_dir_raw, ID, filename)
        plot_cop_overlay(df, output_dir_raw, ID, filename)

        # --- フィルタ・ゼロ平均化 ---
        df_filtered = df.copy()
        df_filtered["ax"] = lowpass_filter(df_filtered["ax"], sampling_rate=1000, cutoff_freq=20)
        df_filtered["ay"] = lowpass_filter(df_filtered["ay"], sampling_rate=1000, cutoff_freq=20)
        df_filtered["ax"] = df_filtered["ax"] - df_filtered["ax"].mean()
        df_filtered["ay"] = df_filtered["ay"] - df_filtered["ay"].mean()
        #plot_cop(df_filtered, output_dir_filtered, ID, filename)
        plot_cop_overlay(df_filtered, output_dir_filtered, ID, filename)

def preparation(ID):
    root_dir = f"D:/User/kanai/Data/{g.datafile}/result_COP/dump/sub{ID+1:02d}*.csv"
    file_list = glob.glob(root_dir, recursive=True)
    return file_list

def output_preparation(subdir):
    output_dir = f"D:/User/kanai/Data/{g.datafile}/result_COP/dump/{subdir}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def lowpass_filter(data, sampling_rate, cutoff_freq):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

"""
サブプロットでX軸とY軸を分けてプロット
"""
def plot_cop(df, output_dir, ID, filename):
    time = df.index / 1000  # assuming the sampling rate is 1000 Hz
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot x-axis vs time
    axs[0].plot(time, df["ax"], label="X-axis")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("X-axis displacement")
    axs[0].set_ylim(-45, 105)
    # Plot y-axis vs time (オレンジ色に変更)
    axs[1].plot(time, df["ay"], label="Y-axis", color="orange")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Y-axis displacement")
    axs[1].set_ylim(-35, 50)

    plt.tight_layout()
    out_name = f"{output_dir}/sub{ID+1}_{filename}.svg"
    plt.savefig(out_name)
    plt.close()



"""
x軸とY軸を同じグラフに重ねてプロット
"""
def plot_cop_overlay(df, output_dir, ID, filename):
    # X軸・Y軸を同じグラフに重ねてプロット
    time = df.index / 1000  # assuming the sampling rate is 1000 Hz
    plt.figure(figsize=(8, 5))
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(time, df["ax"], label="X-axis")
    plt.plot(time, df["ay"], label="Y-axis", color="orange")
    # --- 平均値の赤色点線を追加 ---
    mean_ax = df["ax"].mean()
    mean_ay = df["ay"].mean()
    plt.axhline(mean_ax, color="red", linestyle="dashed", linewidth=1.5)
    plt.axhline(mean_ay, color="red", linestyle="dashed", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylim(-50, 90)
    plt.legend()
    plt.tight_layout()
    out_name = f"{output_dir}/sub{ID+1}_{filename}_overlay.svg"
    plt.savefig(out_name)
    plt.close()

if __name__ == "__main__":
    main()
