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
import global_value as g

for ID in range(g.subnum):
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/sub%d/csv" %(g.datafile, ID+1)
    # 出力先フォルダを作成
    output_dir = os.path.join(root_dir, "fp")
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
                    with open(output_path, "w", newline='') as output_file:
                        writer = csv.writer(output_file)
                        # ヘッダーを出力
                        writer.writerow(["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"])
                        writer.writerow(["TIME","Force_X","Force_Y","Force_Z","Moment_X","Moment_Y","Moment_Z","0","0","COP_X","COP_Y","0","0","0","0"])
                        for i in range(8):
                            writer.writerow([])
                        i = 0
                        for row in reader:
                            # 謎のnanが出るため一番最後のデータを削除
                            if(29995>i):
                                output_row = [row[1],row[2],row[3],row[4],row[5],row[6],row[7],"0","0",row[8],row[9],"0","0","0","0"]
                                writer.writerow(output_row)
                                i = i+1
        # カレントディレクトリの走査が終わったら終了
        break