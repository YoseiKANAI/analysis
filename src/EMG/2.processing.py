# %%
# -*- coding: utf-8 -*-
# バンドパスフィルタのプログラム
"""
Created on: 2025-01-07 00:22

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import signal

import global_value as g

def main():
    sampling = 2000
    for ID in range(g.subnum):
        # ファイルリストを作成
        file_list, output_dir = preparation(ID)
        
        for f in file_list:    
            # 出力ファイル名を作成
            output_filename = f[(f.find("\\")+1):]
            output_path = os.path.join(output_dir, output_filename.replace("_a_2", ""))
            
            df = pd.read_csv(f)
            result = pd.DataFrame()
            
            for i in g.muscle_columns:
                #df_notch = notch(np.array(df[i].interpolate()), sampling)
                df_filter = bandpass(np.array(df[i].interpolate()), sampling)
                result[i] = RMS(df_filter, sampling)
                #result[i] = np.abs(df_filter) 
            
            result.dropna(how="all", axis=1)
            result.to_csv(output_path, index=None)

###
### 筋電データののみのパスリストを作成
###
def preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/*_a_2.csv" %(g.datafile, ID+1)
    file_list = glob.glob(input_dir)
    
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/EMG_proc" %(g.datafile, ID+1)
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    
    return file_list, output_dir

###
### バンドパスフィルタ
###
def bandpass(x, sampling):
    fn = sampling / 2 #ナイキスト周波数
    lowcut=10.0
    highcut=500.0
    low = lowcut/fn
    high = highcut/fn

    b, a = signal.butter(N=4, Wn = [low, high], btype = "band") #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x) #信号に対してフィルタをかける
    return y

###
### ノッチフィルタ　50Hz除去
###
def notch(x, sampling):
    quality_factor = 30.0 #フィルタの品質係数
    fn = sampling / 2 #ナイキスト周波数
    norm_notch_freq = 50.0/fn

    b, a = signal.iirnotch(norm_notch_freq, quality_factor) #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x) #信号に対してフィルタをかける
    return y

###
### 移動RMS
###
def RMS(data, sampling):
    window = int((sampling / 10) / 2)
    slide = int(window / 25)  # スライド幅をウィンドウ幅の1/10に設定
    result = []
    for i in range(0, len(data), slide):
        if i < window:
            result.append(np.sqrt(np.square(data[:i+window]).mean()))
        elif (len(data) - i) < window:
            result.append(np.sqrt(np.square(data[i-window:]).mean()))
        else:
            result.append(np.sqrt(np.square(data[i-window:i+window]).mean()))
    return np.array(result)

if __name__ == "__main__":
    main()
# %%
