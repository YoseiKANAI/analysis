# %%
# -*- coding: utf-8 -*-
# 力覚計算用のデータを格納したcsvを作成
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import csv
import math
import pandas as pd
import savitzky_golay
import global_value as g


# COP入力パス（dump）
root_input_COP = "D:/User/kanai/Data/%s/result_COP/dump" %(g.datafile)

# dumpファイルと力覚を同じディレクトリに移動する
for i in range(g.subnum):
    # ルートフォルダのパスを指定（力覚データの格納場所）
    root_dir = "D:/User/kanai/Data/%s/sub%d/csv" % (g.datafile, i+1)
    
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
            if filename.endswith(("f_1.csv", "f_2.csv","_2D.csv","_6D.csv", "_a.csv", "_a_1.csv", "_a_2.csv")):
                continue
            if filename.startswith(("NC")):
                continue            
            if filename.endswith((".csv")):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)

                # 出力ファイル名を作成
                output_filename = filename
                output_filename = output_filename.replace("00", "")
                output_path_Force = os.path.join(output_dir_Force, output_filename)

                # CSVファイルを開く
                df = pd.read_csv(input_path)
                if "finger X" in df.columns :
                    pass
                else:
                    break
                
                output_row = pd.DataFrame()
                output_row["finger X"] = df["finger X"]
                output_row["finger Y"] = df["finger Y"] 
                output_row["finger Z"] = df["finger Z"]
                
                if filename.startswith(("DW")):
                    output_row["base X"] = df["weight X"]
                    output_row["base Y"] = df["weight Y"]
                    output_row["base Z"] = df["weight Z"]
                else:
                    output_row["base X"] = df["base X"]
                    output_row["base Y"] = df["base Y"]
                    output_row["base Z"] = df["base Z"]
                output_row.to_csv(output_path_Force, index = None)
        # カレントディレクトリの走査が終わったら終了
        break
    
    # COP
    # COPを標準化して比較しやすくする
    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_input_COP):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            sub = "sub%02d" %(i+1)
            if filename.startswith((sub)):
                if filename.endswith((".csv")):
                    # CSVファイルのパスを作成
                    input_path = os.path.join(dirpath, filename)

                    # 出力ファイル名を作成
                    output_filename = filename
                    output_path_COP = os.path.join(output_dir_COP, output_filename)
        
                    # CSVファイルを開く
                    df = pd.read_csv(input_path, names=("COP_X", "COP_Y", "COP"), skiprows = 1)
                    # COP全体の配列を作成
                    for j in range(len(df)):
                        df.iloc[j, 2] = math.sqrt(df.iloc[j, 0]**2 + df.iloc[j, 1]**2)
                    
                    df = savitzky_golay.func(df)
                    # 列ごとに標準化
    #                df_std = pd.DataFrame()
    #                for k in (["COP_X", "COP_Y", "COP"]):
    #                    df_std[k] = (df[k] - df[k].mean()) / df[k].std() 
                    # 列ごとに正規化
                    result = pd.DataFrame()
                    for k in (["COP_X", "COP_Y", "COP"]):
                        result[k] = df[k]  / df[k].max()                  
                    result.to_csv(output_path_COP, index = None)
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