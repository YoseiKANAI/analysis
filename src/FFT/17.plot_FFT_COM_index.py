# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import csv
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import welch
import matplotlib.pyplot as plt
import global_value as g

# sampling_rate[Hz]
f_s = 100

index = ["0~0.2", "0.2~0.4", "0.6~0.8"]
task = ["NC", "DB", "W"]
list = ["COM_X", "COM_Y", "COM"]

for ID in range(g.subnum):
    subID = "sub%d" %(ID+1)

    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/result_FFT/sub%d/COM" %(g.datafile, ID+1)
    
    output_dir = os.path.join( root_dir, "index_COM")
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    
    sum_DB = pd.DataFrame()
    sum_W = pd.DataFrame()
    sum_NC = pd.DataFrame()
    for i in list:
        # ルートフォルダ以下のすべてのフォルダに対して処理を実行
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # フォルダ内のすべてのcsvファイルに対して処理を実行
            for filename in filenames:
                if filename.endswith((".csv")):
                    # CSVファイルのパスを作成
                    input_path = os.path.join(dirpath, filename)
                    
                    # CSVファイルを開く
                    df = pd.read_csv(input_path)
                    
                    sum = pd.DataFrame((df.loc[0:4, i].sum(), df.loc[5:8, i].sum(), df.loc[13:16, i].sum()))
                    
                    # 各データに格納
                    if filename.startswith(("DB")):
                        sum_DB = pd.concat([sum_DB, sum.T])
                    if filename.startswith(("W")):
                        sum_W = pd.concat([sum_W, sum.T])
                    if filename.startswith(("NC")):
                        sum_NC = pd.concat([sum_NC, sum.T])
        
        # タスクごとに指標を算出
        mean = pd.DataFrame([sum_NC.mean(axis = 0), sum_DB.mean(axis = 0), sum_W.mean(axis = 0)])                     
        std = pd.DataFrame([sum_NC.std(axis = 0), sum_DB.std(axis = 0), sum_W.std(axis = 0)]) 
        
        std = std / mean
        mean = mean / mean.iloc[0, :]   
        
        # ラグ
        index_num = np.arange(3)          
        # グラフをプロット
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 24   

        fig = plt.figure(figsize = (12, 8))
        ax = fig.add_subplot(1,1,1)
        
        color = ["tab:blue", "tab:green", "tab:red"]
        
        for t in range(3):
            slide = t*0.25
            err = [std.iloc[t,0], std.iloc[t,1], std.iloc[t,2]]
            ax.bar(index_num+slide, mean.iloc[t, :], width=0.2, capsize=3, label = task[t], color = color[t])
        ax.legend(loc = "upper right", fontsize ="large", ncol=len(task), frameon=False, handlelength = 0.7, columnspacing = 1)
        ax.tick_params(direction="in")
        ax.set_xticks(index_num + 0.25)
        ax.set_ylim([0, 6.0])
        ax.set_xlabel("Freq[Hz]")
        ax.set_ylabel("Power Spectral Density \n normalized By NC values")
        ax.set_xticklabels(index)
        ax.set_title(i)        
#        plt.show()
        plt.savefig(output_dir + "/"+ subID + i + ".png")
        plt.close()