# %%
# # -*- coding: utf-8 -*-
"""
Created on: 2025-03-15 10:58

@author: ShimaLab
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    input_files = {
        "NC": "D:/User/kanai/Data/241223/sub11/csv/EMG_proc/NC0001.csv",
        "FB": "D:/User/kanai/Data/241223/sub11/csv/EMG_proc/FB0002.csv",
        "D1": "D:/User/kanai/Data/241223/sub11/csv/EMG_proc/D10004.csv"
    }
    output_dir = "D:/User/kanai/Data/241223/result_EMG/Raw_plot/proccesed"
    os.makedirs(output_dir, exist_ok=True)
    
    data = {task: pd.read_csv(file) for task, file in input_files.items()}
    plot_emg(data, output_dir, "sub11")

def plot_emg(data, output_dir, filename):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    columns_to_plot = ["TA_R", "SO_R", "PL_R", "MF_R", "IO_L"]
    tasks = ["NC", "FB", "D1"]
    colors = {"NC": "tab:blue", "FB": "tab:orange", "D1": "tab:green"}
    
    for col in columns_to_plot:
        fig, axs = plt.subplots(3, 1, figsize=(4, 4), sharex=True)
        max_ylim = None
        for i, task in enumerate(tasks):
            if col in data[task].columns:
                time = data[task].index / 2000  # assuming the sampling rate is 2000 Hz
                axs[i].plot(time, data[task][col], label=f"{task} - {col}", color=colors[task])
                if task == "NC":
                    max_ylim = data[task][col].max()
                axs[i].set_ylim(0, max_ylim+10)
                #axs[i].set_ylabel("Amplitude")
        #axs[-1].set_xlabel("Time [s]")
        axs[-1].set_xticks([0, time[-1]])  # x軸方向のメモリを0と最後のデータに設定
        axs[-1].set_xticklabels([0, 30])  # メモリの表記を0と30に設定
        plt.tight_layout()
        out_name = f"{output_dir}/{filename}_{col}.svg"
        plt.savefig(out_name)
        plt.close()

if __name__ == "__main__":
    main()