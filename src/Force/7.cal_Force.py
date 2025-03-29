# %%
# -*- coding: utf-8 -*-
# 力覚を算出するプログラム
# フィルタがライブラリのやつ
"""
Created on Mon May  1 16:31:16 2023

@author: ShimaLab
"""
# "C:/Users/ShimaLab/Desktop/one time/test"
import os
import pandas as pd
import savitzky_golay
from scipy.signal import savgol_filter
import global_value as g

# 1フレーム当たりの時間 [s]
d_t = 0.01
# 重力加速度
gravity = 9.806650

list = ["Force X", "Force Y", "Force Z"]

for ID in range(g.subnum):
    # 入力フォルダと出力先フォルダ名を指定
    input_folder = "D:/User/kanai/Data/%s/sub%d/csv/motion" % (g.datafile, ID+1)
    output_folder_name = "Force"

    # 出力先フォルダのパスを作成
    output_folder = os.path.join(input_folder, output_folder_name)

    # 出力先フォルダが存在しない場合は作成する
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 入力フォルダ内のすべてのファイル・フォルダに対して処理を行う
    for root, dirs, files in os.walk(input_folder):
        # 処理対象となるファイルのリストを取得
        file_list = [f for f in files if f.endswith('.csv')]

        # 処理対象となるファイルが存在しない場合は次のフォルダに移る
        if not file_list:
            continue

        # 出力先フォルダのパスを作成
        output_folder_path = os.path.join(output_folder, root[len(input_folder) + 1:])

        # 出力先フォルダが存在しない場合は作成する
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # ファイルごとに処理を行う
        for filename in file_list:
            filepath = os.path.join(root, filename)
            
            task_name = filename[0:2]

            # CSVファイルを読み込む
            df = pd.read_csv(filepath)

            # 各列の1行下との差分を計算
            diff = df.diff().iloc[1:, :]
            
            # 単位をmmからmに補正
            diff = diff * (10**-3)
            # 指先の風船重りに対する相対加速度を計算
            R = (diff.diff(-3, axis=1))/(d_t**2)
            print(df.columns[[3, 4, 5]])
            R = R.drop(columns=df.columns[[3, 4, 5]], axis = 1)
            R = R.rename(columns={"finger X":"Force X", "finger Y":"Force Y", "finger Z":"Force Z"})
            
            # 力覚を算出
            F = R * g.m[task_name] * 10**(-3)
            if filename.startswith("F"):
                F.iloc[:, 2] = F.iloc[:, 2] - g.m[task_name]*gravity*10**(-3)
            else:
                F.iloc[:, 2] = F.iloc[:, 2] + g.m[task_name]*gravity*10**(-3)

            # 出力ファイルパスを指定
            output_filepath = os.path.join(output_folder_path, filename)
            result = pd.DataFrame()
            
            # 列ごとにフィルタをかける
            for i in list: 
                result[i] = savgol_filter(F[i], 5, 2, deriv=0)
            
            # 最大値で正規化
#            result = pd.DataFrame()
#            for k in (["Force X", "Force Y", "Force Z"]):
#                result[k] = F[k]  / F[k].max()                  

            # 差分データをCSVファイルとして出力
            result.to_csv(output_filepath, index=False)
        break

    print("処理が完了しました。")

