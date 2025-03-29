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

list = ["Force X", "Force Y", "Force Z"]

for ID in range(g.subnum):
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/sub%d/csv/motion/Force" %(g.datafile, ID+1)
    output_dir = "D:/User/kanai/Data/%s/result_FFT/sub%d/Force" %(g.datafile, ID+1)
    
    output_dir_gragh = os.path.join(output_dir, "plot")
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_gragh, exist_ok=True)

    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.endswith(("f_1.csv", "f_2.csv","_2D.csv", "_a.csv")):
                continue
            # CSVファイルのパスを作成
            input_path = os.path.join(dirpath, filename)
        
            # 出力ファイル名を作成
            output_filename = filename
            output_path = os.path.join(output_dir, output_filename)
            
            # CSVファイルを開く
            df = pd.read_csv(input_path)
            
            n = len(df)
            
            for i in list:
                # welch法でスペクトル解析
                freqs, psd = welch(df[i], fs = f_s, nperseg=2048)           
                
                # グラフをプロット
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["mathtext.fontset"] = "cm"
                plt.rcParams["font.size"] = 14   
                
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.plot(freqs, psd, label = "freq")
    #            ax.plot(obj["Y"], label = "Y")
    #            ax.plot(obj["Z"], label = "Z")
                ax.legend()
    #                ax.legend(loc = "upper right", fontsize ="large", frameon=False, handlelength = 0.7, columnspacing = 1)
                ax.tick_params(direction="in")
    #            ax.set_xlim([0.0, 1.5])
    #            ax.set_ylim([0.0, 70])
                ax.set_ylabel("freq power")
                ax.set_title(filename.replace(".csv", "") + i)
    #            plt.show()
                fig.savefig(output_dir_gragh + "/" + filename.replace(".csv","") +"_" + i + ".png")
                plt.close()
        # カレントディレクトリの走査が終わったら終了
        break