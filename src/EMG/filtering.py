# %%
# -*- coding: utf-8 -*-
# バンドパスフィルタのプログラム
"""
Created on: 2024-11-26 02:25

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
        output_dir = "D:/User/kanai/Data/%s/sub%d/csv/EMG_filterd" %(g.datafile, ID+1)
        # 出力先フォルダを作成
        os.makedirs(output_dir, exist_ok=True)
        # ファイルリストを作成
        file_list = preparation(ID)
        
        for f in file_list:    
            # 出力ファイル名を作成
            output_filename = f[(f.find("\\")+1):]
            output_path = os.path.join(output_dir, output_filename.replace("_a_2", ""))
            
            df = pd.read_csv(f)
            result = pd.DataFrame()
            window_size = 200
            
            for i in df:
                df_notch = notchpass(np.array(df[i].interpolate()), sampling)
                df_filter = lowpass(df_notch, sampling)
                #result[i] = RMS(df_filter, sampling)
                if "MF" in i:
                    df_filter_detrend = signal.detrend(df_filter)
                    components, singular_values = ssa_decompose(df_filter_detrend, window_size)
                    df_filter = df_filter_detrend - reconstruct_signal_fixed(components, [0])
                if "EO" in i:
                    df_filter_detrend = signal.detrend(df_filter)
                    components, singular_values = ssa_decompose(df_filter_detrend, window_size)
                    df_filter = df_filter_detrend - reconstruct_signal_fixed(components, [0])
                                
                result[i] = df_filter
        
            result.dropna(how="all", axis=1)
            result.to_csv(output_path, index=None)

###
### 筋電データののみのパスリストを作成
###
def preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/*_a_2.csv" %(g.datafile, ID+1)
    file_list = glob.glob(input_dir)
    
    return file_list

###
### ローパスフィルタ
###
def lowpass(x, sampling):
    fn = sampling / 2 #ナイキスト周波数
    highcut=500.0
    high = highcut/fn
    
    b, a = signal.butter(N=4, Wn = high, btype = "low") #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x) #信号に対してフィルタをかける
    return y

###
### ノッチフィルタ　50Hz除去
###
def notchpass(x, sampling):
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
    slide = int((sampling/10) /2)
    result = np.zeros(len(data))
    for i in range(len(data)):
        if i<slide:
            result[i] = np.sqrt(np.square(data[:i+slide]).mean())
        elif (len(data)- i)<slide:
            result[i] = np.sqrt(np.square(data[i-slide:]).mean())
        else:
            result[i] = np.sqrt(np.square(data[i-slide:i+slide]).mean())
    return result

def ssa_decompose(signal, window_size):
    """Singular Spectrum Analysis (SSA) for signal decomposition."""
    n = len(signal)
    K = n - window_size + 1
    trajectory_matrix = np.array([signal[i:i + window_size] for i in range(K)])
    U, S, V = np.linalg.svd(trajectory_matrix, full_matrices=False)
    components = np.dot(U, np.diag(S)).dot(V)
    return components.T, S  # Transpose for easier handling

def reconstruct_signal_fixed(components, indices):
    """Reconstruct the signal using selected components."""
    selected = np.sum(components[indices, :], axis=0)
    return selected

    

if __name__ == "__main__":
    main()
# %%
