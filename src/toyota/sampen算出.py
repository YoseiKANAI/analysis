# -*- coding: utf-8 -*-
# input_folder = "C:/Users/ShimaLab/Desktop/解析用/COP/sub1"

"""
Created on Tue Oct 17 10:50:05 2023

@author: ShimaLab
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_sample_entropy(data, m, r, N):
    # サンプルエントロピーの計算
    # ここにサンプルエントロピーの計算方法を実装する
    # 以下はダミーの例
    return np.random.rand()

def process_csv_file(file_path, m, r, N):
    # CSVファイルの読み込み
    df = pd.read_csv(file_path)

    # 1行目を無視
    df = df.iloc[1:]

    # 列ごとに正規化
    normalized_df = (df - df.min()) / (df.max() - df.min())

    # サンプルエントロピーの計算
    entropy_values = []
    for column in normalized_df.columns:
        entropy_values.append(calculate_sample_entropy(normalized_df[column], m, r, N))

    # 結果をDataFrameにまとめる
    result_df = pd.DataFrame({
        'Column': normalized_df.columns,
        'Sample_Entropy': entropy_values
    })

    return result_df

def main(input_folder, output_file, m, r, N):
    # 出力先フォルダが存在しない場合は作成
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # フォルダ内のCSVファイルを処理
    result_dfs = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            result_dfs.append(process_csv_file(file_path, m, r, N))

    # 結果をまとめる
    final_result_df = pd.concat(result_dfs, ignore_index=True)

    # 結果をCSVファイルに出力
    final_result_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # パラメータの設定
    input_folder = "C:/Users/ShimaLab/Desktop/sampentest"
    output_file = "C:/Users/ShimaLab/Desktop/sampentest/result.csv"
    m = 4  # mの値を指定
    r = 0.25  # rの値を指定
    N = 29700  # Nの値を指定

    # メイン関数の呼び出し
    main(input_folder, output_file, m, r, N)
