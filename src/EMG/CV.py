# %%
# -*- coding: utf-8 -*-
# CVの算出

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
    summary_mean = np.empty(((g.subnum), task_num, g.muscle_num))
    summary_std = np.empty(((g.subnum), task_num, g.muscle_num))
    output_name, output_dir_plot = output_preparation()
    for ID in range(g.subnum):
        filename_list = input_preparation(ID)
        sheet_name = "CV_sub%d" %(ID+1)
        result = create_CV_data(filename_list, ID)
        excel_output(result, output_name, sheet_name)
#        summary(sum, result, ID)
        sub_gragh_plot(result.iloc[26:41, :], output_dir_plot, ID)
#        sub_gragh_plot(result.iloc[69:84, :], output_dir_plot, ID)

        if g.domi_leg[ID]==0:
            result = result.reindex(columns=g.muscle_columns)
        elif g.domi_leg[ID]==1:
            result = result.reindex(columns=["SO_L", "SO_R", "TA_L", "TA_R", "GM_L", "GM_R", "PL_L", "PL_R", "IO_L", "IO_R", "MF_L", "MF_R"])
        
        for m in range(task_num):
            summary_mean[ID, m, :] = result.iloc[m*3+27,:]
            summary_std[ID, m, :] = result.iloc[m*3+28,:]
    muscle = ["SO_domi", "SO_ndomi", "TA_domi", "TA_ndomi", "GM_domi", "GM_ndomi", "PL_domi", "PL_ndomi", "EO_domi", "EO_ndomi", "MF_domi", "MF_ndomi"]
    summary_gragh_plot(summary_mean, summary_std, output_dir_plot, muscle)
            
        

"""
MeanEMGAmplitudeを計算する関数
"""
def create_CV_data(filename_list, ID):
    # 格納するDataFrameを作成
    result = pd.DataFrame(np.full((g.attempt*len(g.task), g.muscle_num), np.nan), columns=g.muscle_columns) 
    # リストの初期化
    task_list = []
    
    # リストの順に呼び出し
    for t in g.task:
        task_list = [s for s in filename_list if t in s]
        # ファイルごとに計算
        for f in task_list:
            df, columns_list = csv_reader(f)
            task = g.task.index(t)
            attempt_num = int(f[(f.find("\\")+6)])
            
            result = cal_CV(df, task, attempt_num, result)
            

    # 平均と分散を導出
    result = cal_mean_std(result, columns_list)
    result_normalization = result.copy()
    
    # NCで正規化したものを算出する
    for i in columns_list:
        result_normalization[i] = result_normalization[i] / result_normalization[i].iloc[27]  
    result_normalization = cal_mean_std(result_normalization, columns_list)
    
    # 空白を追加
    result = pd.concat([result, pd.DataFrame(index=[""])])
    result = pd.concat([result, pd.DataFrame(index=["normalization"])])
    # 生データと結合
    result = pd.concat([result, result_normalization])
    
    return result


"""
CVの計算を行う関数
"""
def cal_CV(df, task, attempt_num, result):
    #CV = pd.DataFrame(index = [index_name], columns=g.muscle_columns)
    CV = df.std(axis = 0, skipna=True).div(df.mean(axis = 0,skipna=True)) * 100
    result.iloc[task*g.attempt + attempt_num-1 :] = CV
    return result

"""
被験者ごとの平均と分散を算出し，格納
"""
def cal_mean_std(CV, columns_list):
    CV = pd.concat([CV, pd.DataFrame(index=[""])])
    # タスクごとの平均と分散を算出
    for i in range(task_num):
        # 空白行を挿入
        CV = pd.concat([CV, pd.DataFrame(index=[g.task[i]])])
        
        # CVからタスクごとのデータを取り出し
        df = np.array(CV.iloc[i*5:(i+1)*5])
        
        # 平均と分散を算出
        mean = pd.DataFrame([np.nanmean(df, axis=0)], columns=columns_list, index=["mean"])
        std = pd.DataFrame([np.nanstd(df, axis=0)], columns=columns_list, index=["std"])
        
        # CVに格納
        CV = pd.concat([CV, mean])
        CV = pd.concat([CV, std])
    return CV


"""
入力パスのリストを作成
"""   
def input_preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" %(g.datafile, ID+1)
    file_list = glob.glob(input_dir)
    
    return file_list         
  
"""
ファイル名の定義
"""        
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CV" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_EMG/CV/plot" %(g.datafile)
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
        err = [result.iloc[i*3+2, :]]
        ax.bar(index_num+slide, result.iloc[i*3+1, :], width=0.17, yerr=err, capsize=3, label = g.task[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0.0, 160])
    ax.set_ylabel("Coefficient of Variantion")
    ax.set_xticklabels(columns_list, rotation = 45)
    #plt.show()
    plot_name = output_dir_plot + "/plot_sub%d.png" %(ID+1)
    fig.savefig(plot_name)

"""
横並びの被験者データを作成
"""
def summary_gragh_plot(mean, std, output_dir, muscle):
    sublist = []
    new_order = [5 ,7, 8, 2, 4, 6]
    for i in new_order:
        """
        if i == 1:
            continue
        """
        sub = "Sub %d" %(i+1)
        sublist.append(sub)
        
    # 特定の群の人のみ抽出
    mean_group = mean[new_order, :, :]
    std_group = std[new_order, :, :]
    sub_num = np.arange(len(new_order))

    for m in range(len(muscle)):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 12 
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for t in range(len(g.task)):
            slide = t*0.15
            err = [std_group[:, t, m]]
            ax.bar(sub_num+slide, mean_group[:, t, m], width=0.12
                , yerr=err, capsize=3, label = g.task[t])
        ax.legend(loc = "upper right", fontsize ="large", ncol=len(g.task), frameon=False, handlelength = 0.7, columnspacing = 1)
        ax.tick_params(direction="in")
        ax.set_xticks(sub_num + 0.3)
        #ax.set_ylim([0, 0.45])
        ax.set_ylabel("Coefficient of Variantion")
        ax.set_xticklabels(sublist)
        #plt.show()
        output_filename = "/plot_%s.png" %(muscle[m])
        fig.savefig(output_dir + output_filename)
        plt.close()

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