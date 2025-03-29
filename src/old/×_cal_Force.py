# %%
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:31:16 2023

@author: ShimaLab
"""
# "C:/Users/ShimaLab/Desktop/one time/test"
import os
import pandas as pd

# 被験者数を指定
subnum = 1

# 1フレーム当たりの時間[ms]
delta_t = 10
# 物体の重さ
m = 5
# 重力加速度
g = 9806650

for i in range(subnum):
    # 入力フォルダと出力先フォルダ名を指定
    input_folder = "D:/User/kanai/Data/test/sub%d/csv/motion" % (i+1)
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

            # CSVファイルを読み込む
            df = pd.read_csv(filepath)

            # 各列の1行下との差分を計算
            diff = df.diff().iloc[1:, :]
            
            # 指先の風船重りに対する相対加速度を計算
            if (filename.startswith("B")):
                Rx = (diff.loc[:, "hand X"]-diff.loc[:,"baloon X"])/(delta_t**2)
                Ry = (diff.loc[:,"hand Y"]-diff.loc[:,"baloon Y"])/(delta_t**2)
            if (filename.startswith("O")):
                Rx = (diff.loc[:,"hand X"]-diff.loc[:,"omori X"])/(delta_t**2)
                Ry = (diff.loc[:,"hand Y"]-diff.loc[:,"omori Y"])/(delta_t**2)
            
            R = pd.concat([Rx, Ry], axis=1)
            
            
            # 力覚を算出
            F = R * m

            # 出力ファイルパスを指定
            output_filepath = os.path.join(output_folder_path, filename)

            # 差分データをCSVファイルとして出力
            F.to_csv(output_filepath, index=False)
        break

    print("処理が完了しました。")

