# %%
# -*- coding: utf-8 -*-

"""
Created on: 2024-12-28 14:55

@author: ShimaLab
"""
import os
import csv
import glob
import global_value as g

def main():
    for ID in range(g.subnum):
        file_list, output_dir = preparation(ID)
        # ファイルごとに計算
        for f in file_list:
            # 出力ファイル名(csvファイル名)を指定
            task_name = f[f.find("\\")+1:]
            output_filename = task_name.replace("tsv", "csv")
            output_path = os.path.join(output_dir, output_filename)
            tsv_to_csv(f, output_path)

"""
入力パスと出力のパスを作成リストを作成
"""
def preparation(ID):
    # フォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/sub%d/" %(g.datafile, ID+1)
    #root_dir = "D:/User/kanai/Data/%s/balloon_test/" %(g.datafile)
    input_dir = root_dir + "*.tsv"
    file_list = glob.glob(input_dir)

    # 出力先フォルダを作成
    output_dir = os.path.join(root_dir, "csv")
    os.makedirs(output_dir, exist_ok=True)

    return file_list, output_dir

"""
tsvをcsvに変換
"""
def tsv_to_csv(f, output_path):
    # TSVファイルを開く
    with open(f, encoding='utf-8', newline='') as input_file:
        tsv = csv.reader(input_file, delimiter = '\t')

        # 出力ファイルを開く
        with open(output_path, "w", newline='') as output_file:
            writer = csv.writer(output_file, delimiter=",")
            i = 0
            # 出力ファイルに書き込み
            # 各型式で上部の余分なデータを除く処理を行う
            for row in tsv:
                if (f.endswith("_2D.tsv")):
                    if (i>6):
                        writer.writerow(row)
                elif (f.endswith("_6D.tsv")):
                    if (i>12):
                        writer.writerow(row)
                elif (f.endswith(("_a_1.tsv", "_a_2.tsv"))):
                    if (i>12):
                        writer.writerow(row)
                elif (f.endswith(("f_1.tsv", "f_2.tsv"))):
                    if (i>25):
                        writer.writerow(row)
                else:
                    if (i>10):
                        writer.writerow(row)
                i = i + 1
            output_file.close()
        input_file.close()

if __name__ == "__main__":
    main()