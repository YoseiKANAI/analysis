# %%
# # -*- coding: utf-8 -*-
"""
Created on: 2024-11-15 13:38

@author: ShimaLab
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats
from scipy.stats import rankdata

import global_value as g

sampling = 2000
task_num = len(g.task)

def main():
    output_dir = "D:/User/kanai/Data/%s/result_EMG/MDF" % (g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    MDF_summary = pd.DataFrame()
    for ID in range(g.subnum):
        # パスを指定
        input_dir = "D:/User/kanai/Data/%s/sub%s/csv/*_a_2.csv" % (g.datafile, ID+1)

        # 各試行全体のMDFを計算
        df = MDF(input_dir, output_name, ID)
        sub = "sub%d" %(ID+1)
        MDF_summary = pd.concat([MDF_summary, pd.DataFrame([sub])])
        MDF_summary = pd.concat([MDF_summary, df])
    sheet_name = "MDF_whole"
    excel_output(MDF_summary, output_name, sheet_name)
        
"""
MDFのメインの関数
"""
def MDF(input_dir, output_name, ID):
    task_list = []
    MDF = pd.DataFrame() 
    MDF_f = pd.DataFrame() 
    MDF_e = pd.DataFrame() 
    sheet_name = "MDF_sub%d" %(ID+1)
    filelist = glob.glob(input_dir)
    for file in filelist:
        attempt = file[35:41]
        task_name = file[35:37]
        task_list.append(task_name)
        df , columns_list = csv_reader(file)
        df_first = df.iloc[:5*sampling,:]
        df_end = df.iloc[25*sampling:,:]

        if (attempt =="FB0001" or attempt =="DW0004") and ID ==0:
            MDF = pd.concat([MDF, pd.DataFrame(index=[task_name])])
            MDF_f = pd.concat([MDF_f, pd.DataFrame(index=[task_name])])
            MDF_e = pd.concat([MDF_e, pd.DataFrame(index=[task_name])])
            continue
        if attempt =="FB0003" and ID ==3:
            MDF = pd.concat([MDF, pd.DataFrame(index=[task_name])])
            MDF_f = pd.concat([MDF_f, pd.DataFrame(index=[task_name])])
            MDF_e = pd.concat([MDF_e, pd.DataFrame(index=[task_name])])
            continue
            
        F = FFT_cal(df)
        F_f= FFT_cal(df_first)
        F_e= FFT_cal(df_end)
        
        N = len(df)
        Ffreq = np.fft.fftfreq(N, d=1/2000)
        index = int(N/(sampling/500))
        
        N_diff = len(df_first)
        Ffreq_diff = np.fft.fftfreq(N, d=1/2000)
        index_diff = int(N_diff/(sampling/500))
        
        MDF = pd.concat([MDF, MDF_cal(F, Ffreq, index, columns_list)])
        MDF_f = pd.concat([MDF_f, MDF_cal(F_f, Ffreq_diff, index_diff, columns_list)])
        MDF_e = pd.concat([MDF_e, MDF_cal(F_e, Ffreq_diff, index_diff, columns_list)])
        
    MDF_diff = MDF_f - MDF_e
    MDF.index = task_list
    MDF_diff.index = task_list
    MDF = pd.concat([pd.DataFrame(index=["生データ"]), MDF])
    MDF = pd.concat([MDF, pd.DataFrame(index=["差分"])])
    MDF = pd.concat([MDF, MDF_diff])
    excel_output(MDF, output_name, sheet_name)
    
    MDF_mean = pd.DataFrame()
    task_list = pd.DataFrame(list(dict.fromkeys(task_list)))
    for i in range(task_num):
        MDF_mean_0 = MDF_diff.iloc[i*5: (i+1)*5].mean()
        MDF_mean = pd.concat([MDF_mean, MDF_mean_0], axis=1)
    result = (MDF_mean.transpose()).reset_index(drop=True)
    result = pd.concat([task_list, result], axis=1)
    return result

"""
csvの読み込みを行う関数
"""
def csv_reader(file):
    df = pd.read_csv(file)
    columns_list = df.columns.values
    return df, columns_list
"""
FFTの計算
"""
def FFT_cal(df):
    f = np.fft.fft(df.interpolate(), axis = 0)
    F = np.abs(f)**2
    return F

"""
MDFを計算, シートに書き込み
"""
def MDF_cal(F, Ffreq, index, columns_list):
    MDF = pd.DataFrame([np.arange(g.muscle_num)], columns = columns_list)
    for muscle in range(g.muscle_num):
        sum_1 = 0
        sum_2 = 0
        cnt = 0
        for i in range(index):
            sum_1 = sum_1 + F[i, muscle]
        for i in range(index):
            sum_2 = sum_2 + F[i, muscle]
            cnt = cnt+1
            if sum_2 >= sum_1/2:
                break
        MDF[columns_list[muscle]] = Ffreq[cnt]
    return MDF

"""
excelに出力
"""
def excel_output(data, output_name, sheet_name):
    if (os.path.isfile(output_name)):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name = sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name = sheet_name)        


if __name__ == "__main__":
    main()