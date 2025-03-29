# %%
# -*- coding: utf-8 -*-
# COPとForceの時系列変化を比較するためのグラフを作成するコード


"""
Created on Wed May  3 18:41:24 2023

@author: ShimaLab
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# 被験者数
subnum = 1

# 出力ファイルのヘッダーを定義する
header = ["File", "type","Lag[ms]", "Correlation"]

# 力覚の名前を定義
type = ["COP_X", "COP_Y", "COP"]
Force = ["Force_X", "Foece_Y"]

for ID in range(subnum):    
    subID = "sub%d" %(ID+1)
        
    # フォルダのパスと出力フォルダのパスを指定する
    input_dir = "D:/User/kanai/Data/240601/sub%d/csv/" %(ID+1)
    output_dir = "D:/User/kanai/Data/240601/result_CAA/"
    os.makedirs(output_dir, exist_ok=True)
    
    # フォルダ内の全てのcsvファイルを読み込む
    COP_files = sorted([f for f in os.listdir(input_dir + "COP/") if f.endswith(".csv")])
    F_files = sorted([f for f in os.listdir(input_dir + "motion/Force/") if f.endswith(".csv")])

    # ファイルごとに処理を行う
    for i, (COP_file, F_file) in enumerate(zip(COP_files, F_files)):
        # result1とresult2のファイルパスを取得する
        COP_file_path = input_dir + 'COP/' + COP_file
        F_file_path = input_dir + 'motion/Force/' + F_file
        
        # COPのデータを読み込む
        with open(COP_file_path) as f:
            df_COP_0 = np.loadtxt(f, delimiter=',', skiprows=1)
            
            df_COP = np.empty([0]) 
            for i in range (0, len(df_COP_0), 10):
                Mid = np.median(df_COP_0[i:i+10,:], axis=0)
                df_COP = np.append(df_COP, Mid)
                
            
            df_COP = df_COP.reshape([-1, 3])    
            df_COP = df_COP[:2940, 2]  # 行数を2940に調整する
            COP_0 = pd.DataFrame(df_COP, columns=["COP"]) 
            COP = (COP_0 - COP_0.mean()) / COP_0.std()
            
        
        # Forceデータを読み込み、列ごとに処理を行う
        with open(F_file_path) as f:
            df_F = np.loadtxt(f, delimiter=',', skiprows=1)
            df_F = df_F[:2940, 0]  # 行数を2940に調整する
            F_0 = pd.DataFrame(df_F, columns=["Force_X"])
            F = (F_0 - F_0.mean()) / F_0.std()


        t = pd.RangeIndex(start=0, stop=2940, step=1)
        plt.figure()
        plt.plot(t, F, label="Force")
        plt.plot(t, COP, label="COP")
        plt.title(F_file)
        plt.legend()
        plt.show()        
#            ax.set_xlabel("Lag")
#            ax.set_ylabel("CCA")
#                plt.savefig("D:/User/kanai/Data/240601/result_CAA/CAAlagplot/" + subID + F_file +"_"+ col_obj +"_"+ col +".png")
#                plt.show()
        plt.close()
        i=i+1