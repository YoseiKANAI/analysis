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
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/COM" %(g.datafile, ID+1)
    
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
                if not "RBHD X" in df.columns:
                    continue
                if not "RBHD X" in df.columns:
                    continue
                if not "RBHD X" in df.columns:
                    continue
                if not "RBHD X" in df.columns:
                    continue
                if not "RELB X" in df.columns:
                    continue
                if not "LELB X" in df.columns:
                    continue
                if not "RWRA X" in df.columns:
                    continue
                if not "LWRA X" in df.columns:
                    continue
                if not "RFIN X" in df.columns:
                    continue
                if not "LFIN X" in df.columns:
                    continue
                
                if not "RTOE X" in df.columns:
                    df["RTOE X"] = df["LHEE X"]
                    df["RTOE Y"] = df["LHEE Y"]
                    df["RTOE Z"] = df["LHEE Z"]
                if not "LHEE X" in df.columns:
                    df["LHEE X"] = df["RTOE X"]
                    df["LHEE Y"] = df["RTOE Y"]
                    df["LHEE Z"] = df["RTOE Z"]
                for i in range(len(df)):
                    if df.loc[i, "RTOE X"] == 0:
                        df["RTOE X"] = df["LHEE X"]
                        df["RTOE Y"] = df["LHEE Y"]
                        df["RTOE Z"] = df["LHEE Z"]
                    if df.loc[i, "LHEE X"] == 0:
                        df["LHEE X"] = df["RTOE X"]
                        df["LHEE Y"] = df["RTOE Y"]
                        df["LHEE Z"] = df["RTOE Z"]
                        
                M_X = pd.DataFrame()
                M_Y = pd.DataFrame()
                COM = pd.DataFrame()

                # 各セグメントのごとのモーメント
                M_X["head_X"] = ((df["RBHD X"]+df["RBHD X"]+df["RBHD X"]+df["RBHD X"])/4 - df["C7 X"]) * 0.5002 * g.weight[ID] * 0.0694
                M_Y["head_Y"] = ((df["RBHD Y"]+df["RBHD Y"]+df["RBHD Y"]+df["RBHD Y"])/4 - df["C7 Y"]) * 0.5002 * g.weight[ID] * 0.0694
                
                M_X["Trunk_X"] = (df["C7 X"]-(df["LASI X"]+df["RASI X"])/2) * 0.4310 * g.weight[ID] * 0.4346
                M_Y["Trunk_Y"] = (df["C7 Y"]-(df["LASI Y"]+df["RASI Y"])/2) * 0.4310 * g.weight[ID] * 0.4346
                
                M_X["R_UpperArm_X"] = (df["RSHO X"] - df["RELB X"]) * 0.5772 * g.weight[ID] * 0.0271
                M_Y["R_UpperArm_Y"] = (df["RSHO Y"] - df["RELB Y"]) * 0.5772 * g.weight[ID] * 0.0271               
                M_X["L_UpperArm_X"] = (df["LSHO X"] - df["LELB X"]) * 0.5772 * g.weight[ID] * 0.0271
                M_Y["L_UpperArm_Y"] = (df["LSHO Y"] - df["LELB Y"]) * 0.5772 * g.weight[ID] * 0.0271
                
                M_X["R_Forearm_X"] = (df["RELB X"] - df["RWRA X"]) * 0.4574 * g.weight[ID] * 0.0162
                M_Y["R_Forearm_Y"] = (df["RELB Y"] - df["RWRA Y"]) * 0.4574 * g.weight[ID] * 0.0162                
                M_X["L_Forearm_X"] = (df["LELB X"] - df["LWRA X"]) * 0.4574 * g.weight[ID] * 0.0162
                M_Y["L_Forearm_Y"] = (df["LELB Y"] - df["LWRA Y"]) * 0.4574 * g.weight[ID] * 0.0162
                
                M_X["R_hand_X"] = (df["RWRA X"] - df["RFIN X"]) * 0.7900 * g.weight[ID] * 0.0061
                M_Y["R_hand_Y"] = (df["RWRA Y"] - df["RFIN Y"]) * 0.7900 * g.weight[ID] * 0.0061                
                M_X["L_hand_X"] = (df["LWRA X"] - df["LFIN X"]) * 0.7900 * g.weight[ID] * 0.0061
                M_Y["L_hand_Y"] = (df["LWRA Y"] - df["LFIN Y"]) * 0.7900 * g.weight[ID] * 0.0061
                
                M_X["R_Thigh_X"] = (df["RASI X"] - df["RKNE X"]) * 0.4095 * g.weight[ID] * 0.1416
                M_Y["R_Thigh_Y"] = (df["RASI Y"] - df["RKNE Y"]) * 0.4095 * g.weight[ID] * 0.1416
                M_X["L_Thigh_X"] = (df["LASI X"] - df["LKNE X"]) * 0.4095 * g.weight[ID] * 0.1416
                M_Y["L_Thigh_Y"] = (df["LASI Y"] - df["LKNE Y"]) * 0.4095 * g.weight[ID] * 0.1416
                
                M_X["R_Shank_X"] = (df["RKNE X"] - df["RANK X"]) * 0.4395 * g.weight[ID] * 0.0433
                M_Y["R_Shank_Y"] = (df["RKNE Y"] - df["RANK Y"]) * 0.4395 * g.weight[ID] * 0.0433
                M_X["L_Shank_X"] = (df["LKNE X"] - df["LANK X"]) * 0.4395 * g.weight[ID] * 0.0433
                M_Y["L_Shank_Y"] = (df["LKNE Y"] - df["LANK Y"]) * 0.4395 * g.weight[ID] * 0.0433
                
                M_X["R_Foot_X"] = (df["RHEE X"] - df["RTOE X"]) * 0.4415 * g.weight[ID] * 0.0137
                M_Y["R_Foot_Y"] = (df["RHEE Y"] - df["RTOE Y"]) * 0.4415 * g.weight[ID] * 0.0137
                M_X["L_Foot_X"] = (df["LHEE X"] - df["LTOE X"]) * 0.4415 * g.weight[ID] * 0.0137
                M_Y["L_Foot_Y"] = (df["LHEE Y"] - df["LTOE Y"]) * 0.4415 * g.weight[ID] * 0.0137
                
                COM["COM_X"] = M_X.sum(axis = 1) / g.weight[ID]
                COM["COM_Y"] = M_Y.sum(axis = 1) / g.weight[ID]
                
                # COP全体の配列を作成
                for j in range(len(COM)):
                    COM.loc[j, "COM"] = math.sqrt(COM.loc[j, "COM_X"]**2 + COM.loc[j, "COM_Y"]**2)
                
                # 正規化 or 標準化
                result = pd.DataFrame()
#                for k in (["COM_X", "COM_Y", "COM"]):
#                    result[k] = COM[k]  / COM[k].max() ×
#                    result[k] = (COM[k] - COM[k].mean()) / COM[k].std()   
                    
                result = COM
                
                # 出力ファイルに力覚データを書き込む
                result.to_csv(output_path, index=False)
                
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
        # カレントディレクトリの走査が終わったら終了
        break