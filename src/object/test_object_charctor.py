# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import math
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import global_value as g

for ID in range(g.subnum):
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/balloon_test/sub%d/csv" %(ID+1)
    output_dir = "D:/User/kanai/Data/balloon_test/sub%d/csv/object" %(ID+1)
    
    output_dir_gragh = os.path.join(output_dir, "plot")
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_gragh, exist_ok=True)

    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.endswith(("f_1.csv", "f_2.csv","_2D.csv", "_a.csv", "_6D.csv")):
                continue
#            if filename.endswith(("_6D.csv")):
            # CSVファイルのパスを作成
            input_path = os.path.join(dirpath, filename)
            # 出力ファイル名を作成
            output_filename = filename
            output_path = os.path.join(output_dir, output_filename)
            
            # 6D用のファイルパスを作成
            filename_6D = filename.replace(".csv", "_6D.csv")
            input_path_6D = os.path.join(dirpath, filename_6D)
                        
            # CSVファイルを開く
            df = pd.read_csv(input_path)
            df_6D = pd.read_csv(input_path_6D)
            obj = pd.DataFrame()
            
            # ファイルの形式に合わせて呼び出し
            taskname = filename[0:2]
            taskname = taskname + " " + "X"
            
            # 物体の座標を抽出
            #"""
            obj["X"] = df_6D[taskname]
            obj["Y"] = df_6D["Y"]
            obj["Z"] = df_6D["Z"]
            
            """
            obj["X"] = df_6D[taskname] - df["base X"]
            obj["Y"] = df_6D["Y"] - df["base Y"]
            obj["Z"] = df_6D["Z"] - df["base Z"]
            """
            
            obj["X"] = obj["X"] / obj["X"].max()
            obj["Y"] = obj["Y"] / obj["Y"].max()
            obj["Z"] = obj["Z"] / obj["Z"].max()
            
#            obj["X"] = (obj["X"] - obj["X"].min())/(obj["X"].max()-obj["X"].min())
#            obj["Y"] = (obj["Y"] - obj["Y"].min())/(obj["Y"].max()-obj["Y"].min())
#            obj["Z"] = (obj["Z"] - obj["Z"].min())/(obj["Z"].max()-obj["Z"].min())
            
            # ピークを検出
            peak, _ = find_peaks(obj["X"], height=[0.50, 0.90], width = [1, 1999], distance = 100)
            
            # 減衰率を計算（3個目まで）
            damp = 0
            sum = 0
            for i in peak:
                if sum == 0:
                    pass
                else:
                    damp = obj["X"].iloc[i]/obj["X"].iloc[past]
                past = i
                sum =sum+1
                if sum == 3:
                    break
        
            damp = math.log(damp)          
            # 振幅を計算
            peak_df = pd.DataFrame(peak)
            amp = (peak_df[0:2].diff()).mean()
                            
            # 出力ファイルに力覚データを書き込む
            obj.to_csv(output_path, index=False)
            
            # グラフをプロット
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams["font.size"] = 16   
            
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(1,1,1)
            ax.plot(obj["X"], label = "X")
#            ax.plot(obj["Y"], label = "Y")
#                ax.plot(obj["Z"], label = "Z")
            ax.plot(peak, obj["X"].iloc[peak],"o",label="peak")
            ax.legend()
#                ax.legend(loc = "upper right", fontsize ="large", frameon=False, handlelength = 0.7, columnspacing = 1)
            ax.tick_params(direction="in")
            ax.set_ylim([-0.1, 1.1])
            ax.set_ylabel("Relative position of \n object and fingertip [mm]")
            ax.set_title(filename.replace(".csv", "") + ": Amp = %f, damp = %f" %(amp, damp))
#                plt.show()
            fig.savefig(output_dir_gragh + "/plot_" + filename.replace(".csv","") + ".png")
            
            """
            # UB,DB    
            if filename.startswith(("DB")):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)
            
                # 出力ファイル名を作成
                output_filename = filename
                output_path = os.path.join(output_dir, output_filename)

                balloon = pd.DataFrame()
                r_ball = pd.DataFrame()
                result  = pd.DataFrame()
                # CSVファイルを開く
                df = pd.read_csv(input_path)
                for i in range(len(df)):
                    if not (df.loc[i, "balloon1 X"] == 0) or (df.loc[i, "balloon6 X"] == 0) or (df.loc[i, "balloon2 X"] == 0) or (df.loc[i, "balloon5 X"] == 0):
                        balloon.loc[i, "X"] = (df.loc[i, "balloon1 X"] + df.loc[i, "balloon6 X"] + df.loc[i, "balloon2 X"] + df.loc[i, "balloon5 X"])/4
                        balloon.loc[i, "Y"] = (df.loc[i, "balloon1 Y"] + df.loc[i, "balloon6 Y"] + df.loc[i, "balloon2 Y"] + df.loc[i, "balloon5 Y"])/4
                        balloon.loc[i, "Z"] = (df.loc[i, "balloon1 Z"] + df.loc[i, "balloon6 Z"] + df.loc[i, "balloon2 Z"] + df.loc[i, "balloon5 Z"])/4
                    elif not (df.loc[i, "balloon3 X"] == 0) or (df.loc[i, "balloon8 X"] == 0) or (df.loc[i, "balloon4 X"] == 0) or (df.loc[i, "balloon7 X"] == 0):
                        balloon.loc[i, "X"] = (df.loc[i, "balloon3 X"] + df.loc[i, "balloon8 X"] + df.loc[i, "balloon4 X"] + df.loc[i, "balloon7 X"])/4
                        balloon.loc[i, "Y"] = (df.loc[i, "balloon3 Y"] + df.loc[i, "balloon8 Y"] + df.loc[i, "balloon4 Y"] + df.loc[i, "balloon7 Y"])/4
                        balloon.loc[i, "Z"] = (df.loc[i, "balloon3 Z"] + df.loc[i, "balloon8 Z"] + df.loc[i, "balloon4 Z"] + df.loc[i, "balloon7 Z"])/4
                    elif not (df.loc[i, "balloon1 X"] == 0) or (df.loc[i, "balloon6 X"] == 0):
                        balloon.loc[i, "X"] = (df.loc[i, "balloon1 X"] + df.loc[i, "balloon6 X"] )/2
                        balloon.loc[i, "Y"] = (df.loc[i, "balloon1 Y"] + df.loc[i, "balloon6 Y"] )/2
                        balloon.loc[i, "Z"] = (df.loc[i, "balloon1 Z"] + df.loc[i, "balloon6 Z"] )/2
                    elif not (df.loc[i, "balloon2 X"] == 0) or (df.loc[i, "balloon5 X"] == 0):
                        balloon.loc[i, "X"] = (df.loc[i, "balloon2 X"] + df.loc[i, "balloon5 X"] )/2
                        balloon.loc[i, "Y"] = (df.loc[i, "balloon2 Y"] + df.loc[i, "balloon5 Y"] )/2
                        balloon.loc[i, "Z"] = (df.loc[i, "balloon2 Z"] + df.loc[i, "balloon5 Z"] )/2  
                    elif not (df.loc[i, "balloon3 X"] == 0) or (df.loc[i, "balloon8 X"] == 0):
                        balloon.loc[i, "X"] = (df.loc[i, "balloon3 X"] + df.loc[i, "balloon8 X"] )/2
                        balloon.loc[i, "Y"] = (df.loc[i, "balloon3 Y"] + df.loc[i, "balloon8 Y"] )/2
                        balloon.loc[i, "Z"] = (df.loc[i, "balloon3 Z"] + df.loc[i, "balloon8 Z"] )/2
                    elif not (df.loc[i, "balloon4 X"] == 0) or (df.loc[i, "balloon7 X"] == 0):
                        balloon.loc[i, "X"] = (df.loc[i, "balloon4 X"] + df.loc[i, "balloon7 X"] )/2
                        balloon.loc[i, "Y"] = (df.loc[i, "balloon4 Y"] + df.loc[i, "balloon7 Y"] )/2
                        balloon.loc[i, "Z"] = (df.loc[i, "balloon4 Z"] + df.loc[i, "balloon7 Z"] )/2
                    else:
                        balloon.loc[i, "X"] = 0
                        balloon.loc[i, "Y"] = 0
                        balloon.loc[i, "Z"] = 0
                
                # 手との相対座標を求める        
                r_ball["X"] = balloon["X"]-df["finger X"]
                r_ball["Y"] = balloon["Y"]-df["finger Y"]
                r_ball["Z"] = balloon["Z"]-df["finger Z"]

                # バターワースローパスフィルタ
                sample_rate = 100
                cutoff_freq = 2
                nyquist_freq = 0.5 * sample_rate
                normal_cutoff = cutoff_freq / nyquist_freq
                b, a = signal.butter(5, normal_cutoff, btype='low')
                
                for k in ("X", "Y", "Z"):
                    r_ball[k] = signal.filtfilt(b, a, r_ball[k])
                    result[k] = (r_ball[k] - r_ball[k].mean()) / r_ball[k].std()    

        
                # 出力ファイルに力覚データを書き込む
                result.to_csv(output_path, index=False)
                
                # グラフをプロット
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["mathtext.fontset"] = "cm"
                plt.rcParams["font.size"] = 14   
                
                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(1,1,1)
                ax.plot(r_ball["X"], label = "X")
                ax.plot(r_ball["Y"], label = "Y")
                ax.plot(r_ball["Z"], label = "Z")
                ax.legend()
#                ax.legend(loc = "upper right", fontsize ="large", frameon=False, handlelength = 0.7, columnspacing = 1)
                ax.tick_params(direction="in")
                ax.set_ylim([-1000, 100])
                ax.set_ylabel("Relative position of \n object and fingertip [mm]")
                ax.set_title(filename.replace(".csv", ""))
#                plt.show()
                fig.savefig(output_dir_gragh + "/plot_" + filename.replace(".csv","") + ".png")
                plt.close()
                """
        # カレントディレクトリの走査が終わったら終了
        break