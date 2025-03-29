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

# フォルダのパスと出力フォルダのパスを指定する
input_dir = "D:/User/kanai/Data/240601/sub1/csv/"
output_dir = "D:/User/kanai/Data/240601/result_CAA/"
os.makedirs(output_dir, exist_ok=True)

# フォルダ内の全てのcsvファイルを読み込む
motion_files = sorted([f for f in os.listdir(input_dir + 'motion/') if f.endswith('.csv')])

# 出力ファイルのヘッダーを定義する
header = ['File', 'Lag[ms]', 'Correlation']

# 手の座標x,y,zに対して行う
for mode in range(3):
    # 出力ファイル名を決定
    if (mode == 0):
        filename = "hand_object_X.csv"
    elif(mode == 1):
        filename = "hand_object_Y.csv"
    else:
        filename = "hand_object_Z.csv"
    
    # 出力ファイルを開く
    with open(output_dir + filename, mode='w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)

        # ファイルごとに処理を行う
        for motion_file in motion_files:
            # motionデータを読み込む
            motion_file_path = input_dir + 'motion/' + motion_file
        
            # motionのデータを読み込み、列ごとに処理を行う
            with open(motion_file_path) as f:
                df = np.loadtxt(f, delimiter=',', skiprows=1)
                df = df[:2940, :]  # 行数を2940に調整する
                # 指先の座標xyz，物体の座標xyz方向それぞれに対して行う
                hand = df[:, mode]
                obj = df[:, mode+3]
                correlation_list = []
                # for k in range(-50, 51):
                #     # ラグを変えながら相互相関解析を行う
                #     correlation = np.corrcoef(result1_column, np.roll(result2_column, k))[0, 1]
                #     correlation_list.append(correlation)
                # max_correlation = max(correlation_list)
                # max_lag = correlation_list.index(max_correlation) - 50
                max_correlation = float('-inf')  # 初期値として無限大を設定
                min_lag = None
                # 相関係数が高くなったタイミングのkを保存する
                # サンプリングが100Hzより1kで10ms
                for k in range(-50, 51):
                # roll関数でobjectをk分スクロールさせる
                    correlation = np.corrcoef(hand, np.roll(obj, k))[0, 1]
                    if correlation > max_correlation:
                        max_correlation = correlation
                        min_lag = k

                lag = min_lag *10
                # 出力ファイルに書き込む
                writer.writerow([motion_file, lag, max_correlation])
