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
type = ["COM_X", "COM_Y", "COM"]
Obj = ["obj_X", "obj_Y"]

corr_list=["obj_X-COM_X", "obj_X-COM_Y", "obj_X-COM", "obj_Y-COM_X", "obj_Y-COM_Y", "obj_Y-COM",]

for ID in range(g.subnum):
    subID = "sub%d" %(ID+1)
    output_filename = subID + ".csv"
            
    # フォルダのパスと出力フォルダのパスを指定する
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/" %(g.datafile, ID+1)
    output_dir = "D:/User/kanai/Data/%s/result_CAA/obj-COM/sub%d/" %(g.datafile, ID+1)
    output_dir_plot = output_dir + "plot/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    
    # フォルダ内の全てのcsvファイルを読み込む
    COM_files = sorted([f for f in os.listdir(input_dir + "COM/") if f.endswith(".csv") and f.startswith(("DB", "W"))])
    obj_files = sorted([f for f in os.listdir(input_dir + "object/") if f.endswith(".csv") and f.startswith(("DB", "W"))])
    
    # 相関係数を記録する配列を定義
    corr_W = np.empty((g.attempt, 6, 2))
    corr_DB = np.empty((g.attempt, 6, 2))
    
    DB_cnt = 0
    W_cnt = 0

    # ファイルごとに処理を行う
    for i, (COM_file, obj_file) in enumerate(zip(COM_files, obj_files)):
        # result1とresult2のファイルパスを取得する
        COM_file_path = input_dir + 'COM/' + COM_file
        obj_file_path = input_dir + 'object/' + obj_file
        
        # COMのデータを読み込む
        with open(COM_file_path) as f:
            df_COM = np.loadtxt(f, delimiter=',', skiprows=1)
            
            df_COM = (df_COM[:2940, :] - df_COM[:2940, :].mean(axis=0)) / df_COM[:2940, :].std(axis=0) # 行数を2940に調整する
            COM = pd.DataFrame(df_COM, columns=type)        
            
        # objデータを読み込み、列ごとに処理を行う
        with open(obj_file_path) as f:
            df_obj = np.loadtxt(f, delimiter=',', skiprows=1)
            df_obj = (df_obj[:2940, 0:2]-df_obj[:2940, 0:2].mean(axis=0)) / df_obj[:2940, 0:2].std(axis=0) # 行数を2940に調整する
            obj = pd.DataFrame(df_obj, columns=Obj)
        
        # modeごとに解析
        for mode in range(2):
            col_obj = Obj[mode]
            obj_file = obj_file.replace(".csv","")

            for i in range(3):
                x = obj.iloc[:, mode]
                y = COM.iloc[:, i]#.values
                                        
                fig = plt.figure(figsize=(6, 4), dpi=120)                
                ax = fig.add_subplot(111)
                xcor_value = ax.xcorr(x, y, maxlags = 400)# xをずらす
                ax.set_ylim([-0.8, 0.8])
                ax.set_title(COM_file.replace(".csv", ""))
                plt.savefig(output_dir_plot + obj_file +"_"+ col_obj +"_"+ type[i] +".png")
#                plt.show()
                plt.close()
                
                # 相関係数の最大値とその時のラグを格納
                corr_ind = pd.DataFrame(xcor_value[1]).idxmax()
                corr_max = np.array([xcor_value[0][corr_ind]*10, xcor_value[1][corr_ind]])
                
                # 相関係数をタスクごとに格納
                if obj_file.startswith("DB"):
                    corr_DB[DB_cnt, mode*3+i] = np.ravel(corr_max)                    
                elif obj_file.startswith("W"):
                    corr_W[W_cnt, mode*3+i] = np.ravel(corr_max)
        if obj_file.startswith("DB"):            
            DB_cnt = DB_cnt+1
        elif obj_file.startswith("W"):
            W_cnt = W_cnt+1
    # ラグと相関係数の分散と平均を求める
    corr_DB_mean = np.mean(corr_DB, axis=0)
    corr_DB_std = np.var(corr_DB, axis=0)
    corr_W_mean = np.mean(corr_W, axis=0)
    corr_W_std = np.var(corr_W, axis=0)
    
    corr = pd.DataFrame(np.concatenate([corr_DB_mean.T, corr_DB_std.T, corr_W_mean.T, corr_W_std.T]), columns = corr_list)   
    output_filepath = os.path.join(output_dir, output_filename)            
    corr.to_csv(output_filepath, index=False)    