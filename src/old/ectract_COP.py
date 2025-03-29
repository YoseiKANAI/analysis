# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""
# input_folder = "D:/toyota/modified/test"
# output_folder = "D:/toyota/modified/test_output"

import os
import csv

# 被験者数を指定
subnum = 1

for i in range(subnum):
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/240601/sub%d/csv" % (i+1)
    # 出力先フォルダを作成
    output_dir = os.path.join(root_dir, "COP")
    os.makedirs(output_dir, exist_ok=True)

    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.endswith(("f_1.csv", "f_2.csv")):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)

                # 出力ファイル名を作成
                output_filename = filename
                output_path = os.path.join(output_dir, output_filename)

                # CSVファイルを開く
                with open(input_path, "r") as input_file:
                    reader = csv.reader(input_file)

                    # ヘッダーをスキップ
                    next(reader)

                    # 出力ファイルを開く
                    with open(output_path, "w", newline="") as output_file:
                        writer = csv.writer(output_file)
                        # ヘッダーを出力
                        writer.writerow(["COP_X", "COP_Y"])
                        # 10,11列目のデータを取得して出力
                        i = 0
                        for row in reader:
                            # 謎のnanが出るため一番最後のデータを削除
                            if (29995>i):
                                output_row = [row[8], row[9]]
                                writer.writerow(output_row)
                                i = i+1
        # カレントディレクトリの走査が終わったら終了
        break
