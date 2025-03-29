# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
import global_value as g

for ID in range(g.subnum):
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/sub%d/csv" %(g.datafile, ID+1)
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/segment_COM" %(g.datafile, ID+1)
    
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
            if filename.endswith((".csv")):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)
            
                # 出力ファイル名を作成
                output_filename = filename
                output_path = os.path.join(output_dir, output_filename)

            
                # CSVファイルを開く
                df = pd.read_csv(input_path)

#                hand = pd.DataFrame()
                COM = pd.DataFrame()

                # 各セグメントのごとのモーメント
                # 体幹
                """
                if "C7 X" and "LASI X" and "RASI X" in df.columns:
                    COM["Trunk_X"] = (df["C7 X"]-(df["LASI X"]+df["RASI X"])/2) * 0.4310
                    COM["Trunk_Y"] = (df["C7 Y"]-(df["LASI Y"]+df["RASI Y"])/2) * 0.4310
                    COM["Trunk_Z"] = (df["C7 Z"]-(df["LASI Z"]+df["RASI Z"])/2) * 0.4310  
                """    
                              
                # 腕
                # 利き手で選択　0：右利き，1：左利き
                if g.sub_domi[ID] == 0:
                    if "RSHO X" and "RELB X" in df.columns:
                        COM["UpperArm_X"] = (df["RSHO X"] - df["RELB X"]) * 0.5772
                        COM["UpperArm_Y"] = (df["RSHO Y"] - df["RELB Y"]) * 0.5772
                        COM["UpperArm_Z"] = (df["RSHO Z"] - df["RELB Z"]) * 0.5772
                    if "RELB X" and "RWRA X" in df.columns:
                        COM["Forearm_X"] = (df["RELB X"] - df["RWRA X"]) * 0.4574
                        COM["Forearm_Y"] = (df["RELB Y"] - df["RWRA Y"]) * 0.4574
                        COM["Forearm_Z"] = (df["RELB Z"] - df["RWRA Z"]) * 0.4574
                    if "RWRA X" and "RFIN X" in df.columns:
                        COM["hand_X"] = (df["RWRA X"] - df["RFIN X"]) * 0.7900
                        COM["hand_Y"] = (df["RWRA Y"] - df["RFIN Y"]) * 0.7900
                        COM["hand_Z"] = (df["RWRA Z"] - df["RFIN Z"]) * 0.7900
                    
                elif g.sub_domi[ID] == 1:
                    if "LSHO X" and "LELB X" in df.columns:
                        COM["UpperArm_X"] = (df["LSHO X"] - df["LELB X"]) * 0.5772
                        COM["UpperArm_Y"] = (df["LSHO Y"] - df["LELB Y"]) * 0.5772
                        COM["UpperArm_Z"] = (df["LSHO Z"] - df["LELB Z"]) * 0.5772
                    if "LELB X" and "LWRA X" in df.columns:
                        COM["Forearm_X"] = (df["LELB X"] - df["LWRA X"]) * 0.4574
                        COM["Forearm_Y"] = (df["LELB Y"] - df["LWRA Y"]) * 0.4574
                        COM["Forearm_Z"] = (df["LELB Z"] - df["LWRA Z"]) * 0.4574
                    if "LWRA X" and "LFIN X" in df.columns:
                        COM["hand_X"] = (df["LWRA X"] - df["LFIN X"]) * 0.7900
                        COM["hand_Y"] = (df["LWRA Y"] - df["LFIN Y"]) * 0.7900
                        COM["hand_Z"] = (df["LWRA Z"] - df["LFIN Z"]) * 0.7900      
    
                # 出力ファイルに力覚データを書き込む
                COM.to_csv(output_path, index=False)
                """"
                # グラフをプロット
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["mathtext.fontset"] = "cm"
                plt.rcParams["font.size"] = 14   
                
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.plot(result["COM_X"], label = "X")
                ax.plot(result["COM_Y"], label = "Y")
                ax.legend()
#                ax.legend(loc = "upper right", fontsize ="large", frameon=False, handlelength = 0.7, columnspacing = 1)
                ax.tick_params(direction="in")
#                ax.set_ylim([0.0, 2.2])
                ax.set_ylabel("COM")
                ax.set_title(filename.replace(".csv", ""))
#                plt.show()
                fig.savefig(output_dir_gragh + "/plot_" + filename.replace(".csv","") + ".png")
                """
        # カレントディレクトリの走査が終わったら終了
        break