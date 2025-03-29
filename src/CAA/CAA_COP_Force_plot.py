# %%
 
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:41:24 2023

@author: ShimaLab
"""

# "C:/Users/ShimaLab/Desktop/one time/result1"
# "C:/Users/ShimaLab/Desktop/one time/result2"
# "C:/Users/ShimaLab/Desktop/one time/output"
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import global_value as g
import statsmodels.api as sm

# 出力ファイルのヘッダーを定義する
header = ["File", "type","Lag[ms]", "Correlation"]

# 力覚の名前を定義
type = ["COP_X", "COP_Y", "COP"]
Force = ["Force_X", "Foece_Y"]

for ID in range(g.subnum):
    for mode in range(2):
        # 出力ファイル名を決定
        if (mode == 0):
            filename = "ForceX_COP.csv"
        elif(mode == 1):
            filename = "ForceY_COP.csv"
        
        subID = "sub%d" %(ID+1)
            
        # フォルダのパスと出力フォルダのパスを指定する
        input_dir = "D:/User/kanai/Data/%s/sub%d/csv/" %(g.datafile, ID+1)
        output_dir = "D:/User/kanai/Data/%s/result_CAA/" %(g.datafile)
        os.makedirs(output_dir, exist_ok=True)
        
        # フォルダ内の全てのcsvファイルを読み込む
        COP_files = sorted([f for f in os.listdir(input_dir + "COP_Standard/") if f.endswith(".csv")])
        F_files = sorted([f for f in os.listdir(input_dir + "motion/Force/") if f.endswith(".csv")])

        # ファイルごとに処理を行う
        for i, (COP_file, F_file) in enumerate(zip(COP_files, F_files)):
            # result1とresult2のファイルパスを取得する
            COP_file_path = input_dir + 'COP_Standard/' + COP_file
            F_file_path = input_dir + 'motion/Force/' + F_file
            
            # COPのデータを読み込む
            with open(COP_file_path) as f:
                df_COP_0 = np.loadtxt(f, delimiter=',', skiprows=1)
                
                df_COP = np.empty([0]) 
                for i in range (len(df_COP_0)):
                    if (i%10 == 0):
                        df_COP = np.append(df_COP, df_COP_0[i, :])
                
                df_COP = df_COP.reshape([-1, 3])    
                df_COP = df_COP[:2940, :]  # 行数を2940に調整する
                COP = pd.DataFrame(df_COP, columns=type)        
            
            # Forceデータを読み込み、列ごとに処理を行う
            with open(F_file_path) as f:
                df_F = np.loadtxt(f, delimiter=',', skiprows=1)
                df_F = df_F[:2940, 0:2]  # 行数を2940に調整する
                F = pd.DataFrame(df_F, columns=Force)

            col_obj = Force[mode]
            cols = type
            plot_size = 500 # プロットするラグの数
            F_file = F_file.replace(".csv","")

            for col in cols:
                x = F.iloc[:, mode]
                y = COP.loc[:, col]#.values
                ccf_xy = sm.tsa.ccf(x, y)[1:plot_size+1]
                ccf_yx = sm.tsa.ccf(y, x)[:plot_size] # (x, y)を基準にしているので、(y, x)はマイナスのラグ 
                ccf = np.concatenate((ccf_yx[::-1], ccf_xy))
        
                x_axis = np.arange(-plot_size, plot_size)
                fig = plt.figure(figsize=(6, 4), dpi=120)
                ax = fig.add_subplot(111)
                ax.plot(x_axis, ccf)
                ax.plot(x_axis, np.zeros(plot_size*2), color = "k", linewidth=0.5)
                ax.set_xlim([-250, 250])
                ax.set_ylim([-0.6, 0.6])
    #            ax.set_title('{} {} vs {}'.format(F_file, col_obj, col))
    #            ax.set_xlabel("Lag")
    #            ax.set_ylabel("CCA")
                plt.savefig("D:/User/kanai/Data/" +g.datafile + "/result_CAA/CAAlagplot/" + subID + F_file +"_"+ col_obj +"_"+ col +".png")
#                plt.show()
                plt.close()
                i=i+1