# %%
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:21:43 2023

@author: ShimaLab
"""
# folder_path = "D:/解析フォルダ/COP/test"


import pandas as pd
import numpy as np
import os
import numpy as np
from math import factorial

# 再帰的にCSVファイルを検索する関数
def search_csv_files(folder_path):
    csv_files = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            csv_files += search_csv_files(item_path)
        elif item.endswith('.csv'):
            csv_files.append(item_path)
    return csv_files

def func(df):
    # サビスキーゴーレイフィルタの重み
    weights = np.array([-2, 3, 6, 7, 8, 7, 6, 3, -2]) / 42
    # 移動平均の窓幅
    rolling_window_size = 30
    
    # 欠損値を0で補完する
    df = df.fillna(0)

    # 各列をサビスキーゴーレイフィルタで平滑化する
    for col_name in df.columns:
        data = df[col_name].values
        smoothed_data = np.convolve(data, weights, mode='same')
        df[col_name] = smoothed_data
    
    # 30点の移動平均を計算する
    for col_name in df.columns:
        data = df[col_name].values
        rolling_mean = pd.Series(data).rolling(window=rolling_window_size, center=False).mean().values
        df[col_name] = rolling_mean
    
    # 空白行を削除して上に詰める
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
