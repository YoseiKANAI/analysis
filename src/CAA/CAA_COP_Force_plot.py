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

# COPデータを読み込む関数
def read_COP_data(cop_file_path):
    df_COP_0 = pd.read_csv(cop_file_path, skiprows=1, header=None).values
    df_COP = np.array([np.median(df_COP_0[i:i+10, :], axis=0) for i in range(0, len(df_COP_0), 10)])
    return df_COP[:2940, :]

# Forceデータを読み込む関数
def read_force_data(force_file_path):
    df_F = pd.read_csv(force_file_path, skiprows=1, header=None).values
    return df_F[:2940, :]

# 相互相関解析を行い、結果をプロットする関数
def compute_and_plot_ccf(F, COP, plot_size, output_path, col_obj, col):
    ccf_xy = sm.tsa.ccf(F, COP)[1:plot_size+1]
    ccf_yx = sm.tsa.ccf(COP, F)[:plot_size]
    ccf = np.concatenate((ccf_yx[::-1], ccf_xy))
    x_axis = np.arange(-plot_size, plot_size)
    
    plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(x_axis, ccf)
    plt.plot(x_axis, np.zeros(plot_size*2), color="k", linewidth=0.5)
    plt.xlim([-250, 250])
    plt.ylim([-0.6, 0.6])
    plt.savefig(output_path)
    plt.close()

# メイン処理を行う関数
def process_data():
    type = ["COP_X", "COP_Y", "COP"]
    Force = ["Force_X", "Force_Y"]
    
    for ID in range(g.subnum):
        input_dir = f"D:/User/kanai/Data/{g.datafile}/sub{ID+1}/csv/"
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
            cop_file_path = input_dir + 'COP_Standard/' + COP_file
            f_file_path = input_dir + 'motion/Force/' + F_file
            
            # COPのデータを読み込む
            COP = read_COP_data(cop_file_path)
            
            # Forceデータを読み込み、列ごとに処理を行う
            F = read_force_data(f_file_path)

            col_obj = Force[mode]
            cols = type
            plot_size = 500 # プロットするラグの数
            F_file = F_file.replace(".csv","")

            for col in cols:
                x = F.iloc[:, mode]
                y = COP.loc[:, col]
                compute_and_plot_ccf(x, y, plot_size, subID, F_file, col_obj, col)
                i=i+1