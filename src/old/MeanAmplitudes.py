# %%
# -*- coding: utf-8 -*-
# 平均EMG振幅の算出
# %MVCは取れていないため，タスク間の比較を行う

"""
Created on: 2024-09-24 17:28

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
    sum = np.zeros((task_num, g.attempt*g.subnum, g.muscle_num))
    output_name, output_dir_plot = prepare()
    for ID in range(g.subnum):
        sheet_name = "MEA_sub%d" %(ID+1)
        input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" %(g.datafile, ID+1)
        result = MeanEMGAmplitude(input_dir, ID)
        excel_output(result, output_name, sheet_name)
#        summary(sum, result, ID)
        sub_gragh_plot(result.iloc[27:42, :], output_dir_plot, ID)
#        sub_gragh_plot(result.iloc[69:84, :], output_dir_plot, ID)
        

"""
MeanEMGAmplitudeを計算する関数
"""
def MeanEMGAmplitude(input_dir, ID):
    # 格納するDataFrameを作成
    MA = pd.DataFrame() 
    # リストの初期化
    task_list = []
    filelist = []
    # 入力ファイル名を入手
    filelist_rand = glob.glob(input_dir)
    for key in g.task:
        out =  [ s for s in filelist_rand if key in s]
        filelist = filelist + out       
    
    # ファイルごとに計算
    for file in filelist:
        df, columns_list = csv_reader(file)
        
        attempt = file[44:50]
        task_name = file[44:46]
        task_list.append(task_name)
        """
        if (attempt =="FB0001" or attempt =="DW0004") and ID ==0:
            MA = pd.concat([MA, pd.DataFrame(index=[task_name])])
            continue
        if attempt =="FB0003" and ID ==3:
            MA = pd.concat([MA, pd.DataFrame(index=[task_name])])
            continue
        """
        raw = pd.DataFrame([df.mean(axis=0)], columns=columns_list, index=[task_name])
        MA = pd.concat([MA, raw])

    # 平均と分散を導出
    result = cal_mean_std(MA, columns_list, task_list)
    
    # NCで正規化したものを算出する
    for i in columns_list:
        MA[i] = MA[i] / result[i].iloc[27]  
    result_normalization = cal_mean_std(MA, columns_list, task_list)
    
    # 空白を追加
    result = pd.concat([result, pd.DataFrame(index=[""])])
    result = pd.concat([result, pd.DataFrame(index=["normalization"])])
    # 生データと結合
    result = pd.concat([result, result_normalization])
    
    return result
        
           
"""
ファイル名の定義
"""        
def prepare():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/D;ata/%s/result_EMG/MeanAmplitude" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_EMG/MeanAmplitude/plot" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_name, output_dir_plot

"""
csvの読み込みを行う関数
"""
def csv_reader(file):
    df = pd.read_csv(file)
    columns_list = df.columns.values
    return df, columns_list

"""
被験者ごとの平均と分散を算出し，格納
"""
def cal_mean_std(MA, columns_list, task_list):
    MA = pd.concat([MA, pd.DataFrame(index=[""])])
    # タスクごとの平均と分散を算出
    for i in range(task_num):
        # 空白行を挿入
        MA = pd.concat([MA, pd.DataFrame(index=[task_list[i*5]])])
        
        # MAからタスクごとのデータを取り出し
        df = np.array(MA.iloc[i*5:(i+1)*5])
        
        # 平均と分散を算出
        mean = pd.DataFrame([np.nanmean(df, axis=0)], columns=columns_list, index=["mean"])
        std = pd.DataFrame([np.nanstd(df, axis=0)], columns=columns_list, index=["std"])
        
        # MAに格納
        MA = pd.concat([MA, mean])
        MA = pd.concat([MA, std])
    return MA

"""
全体のデータをsumにまとめる
"""
def summary(sum, result, ID):
    for i in range(task_num):#タスク数
        for j in range(g.attempt):#試行数
            for k in range(g.muscle_num):#筋電数
                sum[i, (ID*g.attempt)+j, k] = result.iloc[(i*g.attempt)+j+43, k]
    return sum

"""
被験者ごとのグラフをプロットする関数
"""
def sub_gragh_plot(result, output_dir_plot, ID):
    columns_list = result.columns.values
    index_num = np.arange(g.muscle_num)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12   

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(task_num):       
        slide = i*0.17
        err = [result.iloc[i*3+1, :]]
        ax.bar(index_num+slide, result.iloc[i*3, :], width=0.17, yerr=err, capsize=3, label = g.task[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0.0, 0.4])
    ax.set_ylabel("Average values normalized \n By NC values")
    ax.set_xticklabels(columns_list, rotation = 45)
    #plt.show()
    plot_name = output_dir_plot + "/plot_sub%d.png" %(ID+1)
    fig.savefig(plot_name)

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