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
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import global_value as g

for ID in range(g.subnum):
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/sub%d/csv" %(g.datafile, ID+1)
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/object" %(g.datafile, ID+1)
    
    output_dir_gragh = os.path.join(output_dir, "plot")
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_gragh, exist_ok=True)

    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.endswith(("f_1.csv", "f_2.csv","_2D.csv", "_a.csv", "_6D.csv", "_a_1.csv", "_a_2.csv")):
                continue
            if filename.startswith("NC"):
                continue
#            if filename.endswith(("_6D.csv")):
            # CSVファイルのパスを作成
            input_path = os.path.join(dirpath, filename)
            # 出力ファイル名を作成
            output_filename = filename
            output_path = os.path.join(output_dir, output_filename)
            

                        
            # CSVファイルを開く
            df = pd.read_csv(input_path)
            obj = pd.DataFrame()
            
            # fingerが取れていない場合はその試行をスキップ
            if "finger X" in df.columns:
                pass
            else:
                continue
            
            if filename.startswith(("DW")):
                """
                obj["X"] = df["weight X"]
                obj["Y"] = df["weight Y"]
                obj["Z"] = df["weight Z"]
                
                """
                obj["X"] = df["weight X"] - df["finger X"]
                obj["Y"] = df["weight Y"] - df["finger Y"]
                obj["Z"] = df["weight Z"] - df["finger Z"]
            
            else:
                # 6D用のファイルパスを作成
                filename_6D = filename.replace(".csv", "_6D.csv")
                input_path_6D = os.path.join(dirpath, filename_6D)
                df_6D = pd.read_csv(input_path_6D)
                # ファイルの形式に合わせて呼び出し
                taskname = filename[0:2]
                taskname = taskname + " " + "X"
                
                # 物体の座標を抽出
                """
                obj["X"] = df_6D[taskname]
                obj["Y"] = df_6D["Y"]
                obj["Z"] = df_6D["Z"]
                
                """
                obj["X"] = df_6D[taskname] - df["finger X"]
                obj["Y"] = df_6D["Y"] - df["finger Y"]
                obj["Z"] = df_6D["Z"] - df["finger Z"]
                
            """
            obj["X"] = obj["X"] / obj["X"].max()
            obj["Y"] = obj["Y"] / obj["Y"].max()
            obj["Z"] = obj["Z"] / obj["Z"].max()
            """
            
#            obj["X"] = (obj["X"] - obj["X"].min())/(obj["X"].max()-obj["X"].min())
#            obj["Y"] = (obj["Y"] - obj["Y"].min())/(obj["Y"].max()-obj["Y"].min())
#            obj["Z"] = (obj["Z"] - obj["Z"].min())/(obj["Z"].max()-obj["Z"].min())
                            
            # 出力ファイルに力覚データを書き込む
            obj.to_csv(output_path, index=False)
            
            # グラフをプロット
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams["font.size"] = 14   
            
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(1,1,1)
            ax.plot(obj["X"], label = "X")
            ax.plot(obj["Y"], label = "Y")
#            ax.plot(obj["Z"], label = "Z")
            ax.legend()
#                ax.legend(loc = "upper right", fontsize ="large", frameon=False, handlelength = 0.7, columnspacing = 1)
            ax.tick_params(direction="in")
            ax.set_ylim([-160, 160])
            ax.set_ylabel("Relative position of \n object and fingertip [mm]")
            ax.set_title(filename.replace(".csv", ""))
#                plt.show()
            fig.savefig(output_dir_gragh + "/plot_" + filename.replace(".csv","") + ".png")
            plt.close()
        # カレントディレクトリの走査が終わったら終了
        break