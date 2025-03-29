# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import csv
import pandas as pd
import savitzky_golay
from sklearn.preprocessing import StandardScaler
import math

# 被験者数を指定
subnum = 3


# COP入力パス（dump）
root_input_COP = "D:/User/kanai/Data/test/result_COP/dump"

# dumpファイルと力覚を同じディレクトリに移動する
for ID in range(subnum):
    #被験者
    subID = "sub%02d" %(ID+1)
    # ルートフォルダのパスを指定（力覚データの格納場所）
    root_dir = "D:/User/kanai/Data/test/sub%d/csv" % (ID+1)
    
    # 出力先フォルダを作成
    output_dir_Force = os.path.join(root_dir, "motion")
    os.makedirs(output_dir_Force, exist_ok=True)
    
    # 出力先フォルダを作成
    output_dir_COP = os.path.join(root_dir, "COP_Standard")
    os.makedirs(output_dir_COP, exist_ok=True)

    # 力覚
    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.endswith(("f_1.csv", "f_2.csv","_2D.csv", "_a.csv")):
                continue
            if filename.startswith(("NC", "HAJI")):
                continue            
            if filename.endswith((".csv")):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)

                # 出力ファイル名を作成
                output_filename = filename
#                output_filename = output_filename.replace("00", "")
                output_path_Force = os.path.join(output_dir_Force, output_filename)

                # CSVファイルを開く
                df = pd.read_csv(input_path)
                result = df.iloc[:,2:8]
                
                # savitzky_golayフィルタをかける
#                result = savitzky_golay.func(df)

                # 出力ファイルに力覚データを書き込む
                result.to_csv(output_path_Force, index=False)
        # カレントディレクトリの走査が終わったら終了
        break
    
    # COP
    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_input_COP):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.startswith((subID)):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)

                # 出力ファイル名を作成
                output_filename = filename
                output_path_COP = os.path.join(output_dir_COP, output_filename)
    
                # CSVファイルを開く
                df = pd.read_csv(input_path, names = ("COP_X", "COP_Y", "COP"), skiprows = 1)
#                df = df.iloc[: ,0:2]
                
                # savitzky_golayフィルタをかける
                result = savitzky_golay.func(df)
                
                # COP全体の配列を作成
                for i in range(len(result)):
                    result.iloc[i, 2] = math.sqrt(result.iloc[i, 0]**2 + result.iloc[i, 1]**2)
                
                # 出力ファイルに力覚データを書き込む
                result.to_csv(output_path_COP, index=False)

        # カレントディレクトリの走査が終わったら終了
        break
    """
    データ数を合わせる場合
                        # 出力ファイルにCOPデータを書き込む
                        # フォースプレート：1kHz
                        # モーキャプ：0.1kHz
                        # サンプリングを合わせるため10刻みで記入
                        for row in reader:
                            if ((i+1)%10 == 1):
                                output_row = [row[0], row[1]]
                                writer.writerow(output_row)
                                i = i+1
                            else:
                                i = i+1
    
    """