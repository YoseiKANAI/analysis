# %%
# -*- coding: utf-8 -*-
# COP軌跡の導出

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import global_value as g

# ルートフォルダのパスを指定
root_dir = "D:/User/kanai/Data/%s/result_COP/dump" %(g.datafile)
output_dir = "D:/User/kanai/Data/%s/result_COP/dump/plot" %(g.datafile)

# 出力先フォルダを作成
os.makedirs(output_dir, exist_ok=True)

def lowpass_filter(data, sampling_rate, cutoff_freq):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def plot_cop_xy(x, y, out_path, title, xlim=(-20,82), ylim=(-32,42), mean_marker=True):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 26

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, linewidth=1)
    # --- 原点を強調表示（真ん中が空いた黒丸、黒縁を太く）---
    ax.plot(0, 0, marker='o', markerfacecolor='white', markeredgecolor='black', markersize=14, markeredgewidth=4, label='Origin')
    # --- 原点から各軸へ点線を伸ばす ---
    ax.axhline(0, color='black', linestyle='dashed', linewidth=1)
    ax.axvline(0, color='black', linestyle='dashed', linewidth=1)
    if mean_marker:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        ax.plot(mean_x, mean_y, marker='x', color='red', markersize=16, markeredgewidth=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(direction="in")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)

def process_subject11(root_dir, output_dir):
    sampling_rate = 1000
    cutoff_freq = 20
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if not file_name.startswith("sub11"):
                continue
            file_path = os.path.join(root, file_name)
            df = pd.read_csv(file_path)
            x = df["ax"]
            y = df["ay"]
            x_raw = x
            y_raw = y
            # --- ゼロ平均化 ---
            x = x - np.mean(x)
            y = y - np.mean(y)
            x_filt = lowpass_filter(x, sampling_rate, cutoff_freq)
            y_filt = lowpass_filter(y, sampling_rate, cutoff_freq)

            # 生データプロット
            plot_cop_xy(
                x_raw, y_raw,
                out_path=output_dir + f"/plot_{file_name.replace('.csv', '')}_raw.svg",
                title=f"Raw Plot of {file_name.replace('.csv', '')}"
            )
            # フィルタ後データプロット
            plot_cop_xy(
                x_filt, y_filt,
                out_path=output_dir + f"/plot_{file_name.replace('.csv', '')}_filtered.svg",
                title=f"Filtered Plot of {file_name.replace('.csv', '')}"
            )

"""
# グラフをプロット
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 18

fig, ax = plt.subplots(figsize=(10, 8))

for file_name, (x, y) in data_dict.items():
    ax.plot(x, y, linewidth=0.5, label=file_name.replace(".csv", ""))

ax.set_xlabel('X Direction')
ax.set_ylabel('Y Direction')
ax.set_xlim(0, 105)
ax.set_ylim(-25, 50)
ax.tick_params(direction="in")

fig.suptitle("Comparison of sub11FB02 and sub11NC02")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

fig.savefig(output_dir + "/plot_comparison_sub11FB02_sub11NC02.svg")
plt.close(fig)
"""

if __name__ == "__main__":
    process_subject11(root_dir, output_dir)