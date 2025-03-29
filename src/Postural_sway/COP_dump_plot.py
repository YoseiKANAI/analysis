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

# 特定のファイル名を指定
target_files = ["sub11FB02.csv", "sub11NC02.csv"]

data_dict = {}

for root, dirs, files in os.walk(root_dir):
    for file_name in files:
        if "sub11" in file_name:
            # CSVファイルを読み込む
            file_path = os.path.join(root, file_name)
            df = pd.read_csv(file_path)
            
            # 50Hzのローパスフィルタを適用
            sampling_rate = 1000  # サンプリングレートを適切に設定してください
            cutoff_freq = 50
            
            #x = df["ax"]
            #y = df["ay"]
            
            x = lowpass_filter(df["ax"], sampling_rate, cutoff_freq)
            y = lowpass_filter(df["ay"], sampling_rate, cutoff_freq)            
            
            # 各ファイルについて個別にプロット
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams["font.size"] = 26   
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1,1,1) 
            
            ax.plot(x, y, linewidth=1, label=file_name.replace(".csv", ""))
            ax.set_xlabel('X Direction')
            ax.set_ylabel('Y Direction')
            ax.set_xlim(0, 105)
            ax.set_ylim(-25, 50)
            ax.tick_params(direction="in")
            fig.suptitle(f"Plot of {file_name.replace('.csv', '')}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(output_dir + f"/plot_{file_name.replace('.csv', '')}.svg")
            plt.close(fig)

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