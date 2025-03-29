# %%
# -*- coding: utf-8 -*-
# サンプルエントロピーを算出するコード
"""
Created on: 2025-01-04 19:34

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from multiprocessing import Pool

import global_value as g

sampling = 2000

def main():
    summary = np.empty((len(g.task), (g.subnum-1)*g.attempt, g.muscle_num))
    output_name, output_dir_plot = output_preparation()
    for ID in range(g.subnum):
        if ID == 1:
            continue
        # 出力先フォルダを作成
        filename_list = input_preparation(ID)
        sheet_name = "SE_sub%d" %(ID+1)
        
        result = pd.DataFrame(columns = g.muscle_columns)
        result = create_individual_data(result, filename_list)
        excel_output(result, output_name, sheet_name)

def sample_entropy(time_series, m, r):
    N = len(time_series)
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        x = np.array([time_series[i:i + m] for i in range(N - m + 1)], dtype=np.float32)
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)
    
    return -np.log(_phi(m + 1) / _phi(m))

def calculate_sample_entropy_chunk(chunk, m, r):
    if len(chunk) >= m + 1:
        return sample_entropy(chunk, m, r)
    return None

def calculate_sample_entropy_in_chunks_parallel(time_series, m, r):
    chunk_size = 500
    num_chunks = len(time_series) // chunk_size
    chunks = [time_series[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    with Pool() as pool:
        entropies = pool.starmap(calculate_sample_entropy_chunk, [(chunk, m, r) for chunk in chunks])
    
    entropies = [e for e in entropies if e is not None]
    return np.mean(entropies)

# 使用例:
# emg_signalがEMG信号データを含むnumpy配列であると仮定します
# emg_signal = np.array([...])
# m = 2  # 比較するシーケンスの長さ
# r = 0.2 * np.std(emg_signal)  # 許容範囲（信号の標準偏差の20%）

# sample_entropy_value = sample_entropy(emg_signal, m, r)
# print(f"Sample Entropy: {sample_entropy_value}")

def create_individual_data(result, filename_list):
    """
    resultに被験者ごとのデータを格納する
    """
    for t in g.task:
        task_list = [s for s in filename_list if t in s]
        for f in task_list:
            df = pd.read_csv(f)
            # 出力ファイル名を作成
            taskname = f[(f.find("\\")+1):(f.find("\\")+3)]
            attempt_num = int(f[(f.find("\\")+6)])
            index_name = taskname + "_%s" %(attempt_num)
            
            entropy = pd.DataFrame(index = [index_name], columns=g.muscle_columns)
            for m in df:
                r = 0.2 * df[m].std()
                entropy[m] = calculate_sample_entropy_in_chunks_parallel(np.array((df[m]), dtype="float32"), 2, r)
            result = pd.concat([result, entropy.astype(np.float64)])
    return result


def cal_mean_std(result):
    """
    平均と分散を算出する関数
    """
    mean = pd.DataFrame(index=g.task, columns=g.muscle_columns)
    std = pd.DataFrame(index=g.task, columns=g.muscle_columns)
    
    for t in range(len(g.task)):
        mean.iloc[t, :] = result.iloc[t*g.attempt:(t+1)*g.attempt, :].mean()
        std.iloc[t, :] = result.iloc[t*g.attempt:(t+1)*g.attempt, :].std()
        
    return mean, std


def input_preparation(ID):
    """
    入力パスのリストを作成
    """
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" %(g.datafile, ID+1)
    file_list = glob.glob(input_dir)
    
    return file_list

     
def output_preparation():
    """
    ファイル名の定義
    """   
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/sample_entropy" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_name,


def excel_output(data, output_name, sheet_name):
    """
    excelに出力
    """
    if (os.path.isfile(output_name)):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name = sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name = sheet_name)
            
if __name__ == "__main__":
    main()
# %%
