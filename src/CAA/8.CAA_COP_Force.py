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
import csv
import numpy as np
import global_value as g

for ID in range(g.subnum):
    # 出力ファイルのヘッダーを定義する
    header = ["File", "type","Lag[ms]", "Correlation"]
    # 力覚の名前を定義
    type = ["COP_X", "COP_Y", "COP"]


    # フォルダのパスと出力フォルダのパスを指定する
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/" %(g.datafile, ID+1)
    output_dir = "D:/User/kanai/Data/%s/result_CAA/" %(g.datafile)
    os.makedirs(output_dir, exist_ok=True)

    # フォルダ内の全てのcsvファイルを読み込む
    COP_files = sorted([f for f in os.listdir(input_dir + "COP_Standard/") if f.endswith(".csv")])
    F_files = sorted([f for f in os.listdir(input_dir + "motion/Force/") if f.endswith(".csv")])

    # 出力ファイルのヘッダーを定義する
    header = ["File", "type","Lag[ms]", "Correlation"]

    # 力覚の名前を定義
    type = ["COP_X", "COP_Y", "COP"]

    for mode in range(3):
        # 出力ファイル名を決定
        if (mode == 0):
            filename = "sub%d_ForceX_COP.csv" %(ID+1)
        elif(mode == 1):
            filename = "sub%d_ForceY_COP.csv" %(ID+1)
        else:
            filename = "sub%d_ForceZ_COP.csv" %(ID+1)
        # 出力ファイルを開く
        with open(output_dir + filename, mode='w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)

            # ファイルごとに処理を行う
            for i, (COP_file, F_file) in enumerate(zip(COP_files, F_files)):
                # result1とresult2のファイルパスを取得する
                COP_file_path = input_dir + 'COP_Standard/' + COP_file
                F_file_path = input_dir + 'motion/Force/' + F_file
            
                # COPのデータを読み込む
                with open(COP_file_path) as f:
                    df_COP_0 = np.loadtxt(f, delimiter=',', skiprows=1)
                
                    df_COP = np.empty([0]) 
                    for i in range (0, len(df_COP_0), 10):
                        Mid = np.median(df_COP_0[i:i+10,:], axis=0)
                        df_COP = np.append(df_COP, Mid)
                
                    df_COP = df_COP.reshape([-1, 3])    
                    df_COP = df_COP[:2940, :]  # 行数を2940に調整する        
            
                # Forceデータを読み込み、列ごとに処理を行う
                with open(F_file_path) as f:
                    df_F = np.loadtxt(f, delimiter=',', skiprows=1)
                    df_F = df_F[:2940, :]  # 行数を2940に調整する
                    
                    for j in range(3):
                        F = df_F[:, mode]
                        COP = df_COP[:, j]
                        correlation_list = []
                    # for k in range(-50, 51):
                        # ラグを変えながら相互相関解析を行う
                    #     correlation = np.corrcoef(result1_column, np.roll(result2_column, k))[0, 1]
                    #     correlation_list.append(correlation)
                    # max_correlation = max(correlation_list)
                    # max_lag = correlation_list.index(max_correlation) - 50
                        max_correlation = float('-inf')  # 初期値として無限大を設定
                        min_lag = None
                        # 相関係数が高くなったタイミングのkを保存する
                        # サンプリングが100Hzより1kで10ms
                        for k in range(-50, 51):
                        # roll関数でForceをk分スクロールさせる
                            correlation = np.corrcoef(F, np.roll(COP,k))[0, 1]
                            if correlation > max_correlation:
                                max_correlation = correlation
                                min_lag = k

                        lag = min_lag *10
                    
                        if (j == 0):
                            writer.writerow([F_file, type[j], lag, max_correlation])
                        else:
                            writer.writerow(["", type[j], lag, max_correlation])
