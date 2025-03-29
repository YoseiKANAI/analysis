# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import welch
import matplotlib.pyplot as plt
import global_value as g

def main():
    # sampling_rate[Hz]
    f_s = 1000
    output_name, output_dir_plot = output_preparation()
    column_list = ["vision_X", "vestibular_sense_X", "proprioception_X", "vision_Y", "vestibular_sense_Y", "proprioception_Y"]
    summary_mean = np.empty(((g.subnum), len(g.task), len(column_list)))
    summary_std = np.empty(((g.subnum), len(g.task), len(column_list)))

    for ID in range(g.subnum):
        # ルートフォルダのパスを指定
        file_list = preparation()
        sheet_name = "FFT_sub%d" %(ID+1)
        sub_name = "sub0%d" %(ID+1)
        result = pd.DataFrame(columns=column_list)

        # リストの順に呼び出し
        sub_list = [s for s in file_list if sub_name in s]
        for t in g.task:
            task_list = [s for s in sub_list if t in s]
            for f in task_list:
                # CSVファイルを開く
                df = pd.read_csv(f)
                attempt_num = int(f[(f.find("\\")+9)])
                index_name = t + "_%s" %(attempt_num)
                                    
                FFT_index = pd.DataFrame([np.arange(6)],index = [index_name], columns = column_list)
                
                # welch法でスペクトル解析
                for i in range(2):
                    freqs, psd = welch(df.iloc[:, i], fs = f_s, nperseg=4096)
                    if i==0:
                        FFT_index["vision_X"] = np.sum(psd[0:2])
                        FFT_index["vestibular_sense_X"] = np.sum(psd[2:5])
                        FFT_index["proprioception_X"] = np.sum(psd[5:13])
                    if i==1:
                        FFT_index["vision_Y"] = np.sum(psd[0:2])
                        FFT_index["vestibular_sense_Y"] = np.sum(psd[2:5])
                        FFT_index["proprioception_Y"] = np.sum(psd[5:13])                  
                result = pd.concat([result, FFT_index])
                    
        # 平均と分散を導出
        result = cal_mean_std(result, column_list)
        excel_output(result, output_name, sheet_name)
        sub_gragh_plot(result.iloc[27:42, :], output_dir_plot, ID) 
        for m in range(len(g.task)):
            summary_mean[ID, m, :] = result.iloc[m*3+27,:]
            summary_std[ID, m, :] = result.iloc[m*3+28,:]
    summary_mean = summary_mean[[0,2,3,4,5], :, :]
    summary_std = summary_std[[0,2,3,4,5], :, :]
    summary_gragh_plot(summary_mean, summary_std, output_dir_plot, column_list)



"""
COPファイルのパスリストを作成
"""
def preparation():
    # フォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/result_COP/dump/*.csv" %(g.datafile)
    file_list = glob.glob(root_dir)
    
    return file_list

"""
ファイル名の定義
"""        
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_FFT/COP" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_FFT/COP/plot" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_name, output_dir_plot

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

"""
被験者ごとの平均と分散を算出し，格納
"""
def cal_mean_std(data, columns_list):
    data = pd.concat([data, pd.DataFrame(index=[""])])
    # タスクごとの平均と分散を算出
    for i in range(len(g.task)):
        # 空白行を挿入
        data = pd.concat([data, pd.DataFrame(index=[g.task[i]])])
        
        # dataからタスクごとのデータを取り出し
        df = np.array(data.iloc[i*5:(i+1)*5])
        
        # 平均と分散を算出
        mean = pd.DataFrame([np.nanmean(df, axis=0)], columns=columns_list, index=["mean"])
        std = pd.DataFrame([np.nanstd(df, axis=0)], columns=columns_list, index=["std"])
        
        # dataに格納
        data = pd.concat([data, mean])
        data = pd.concat([data, std])
    return data

"""
被験者ごとのグラフをプロットする関数
"""
def sub_gragh_plot(result, output_dir_plot, ID):
    columns_list = result.columns.values
    index_num = np.arange(len(columns_list))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12   

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(len(g.task)):       
        slide = i*0.17
        err = [result.iloc[i*3+1, :]]
        ax.bar(index_num+slide, result.iloc[i*3, :], width=0.17, yerr=err, capsize=3, label = g.task[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=len(g.task), frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
#    ax.set_ylim([0.0, 0.12])
    ax.set_ylabel("FFT")
    ax.set_xticklabels(columns_list, rotation = 45)
    #plt.show()
    plot_name = output_dir_plot + "/plot_sub%d.png" %(ID+1)
    fig.savefig(plot_name)     


"""
横並びの被験者データを作成
"""
def summary_gragh_plot(mean, std, output_dir, muscle):
    sublist = []
    for i in range(g.subnum):
        if i == 1:
            continue
        sub = "Sub %d" %(i+6)
        sublist.append(sub)
    
    sub_num = np.arange(g.subnum-1)

    for m in range(len(muscle)):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 12 
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for t in range(len(g.task)):
            slide = t*0.15
            err = [std[:, t, m]]
            ax.bar(sub_num+slide, mean[:, t, m], width=0.12
                , yerr=err, capsize=3, label = g.task[t])
        ax.legend(loc = "upper right", fontsize ="large", ncol=len(g.task), frameon=False, handlelength = 0.7, columnspacing = 1)
        ax.tick_params(direction="in")
        ax.set_xticks(sub_num + 0.3)
        #ax.set_ylim([0, 0.05])
        ax.set_ylabel("Coefficient of Variantion")
        ax.set_xticklabels(sublist)
        #plt.show()
        output_filename = "/plot_%s.png" %(muscle[m])
        fig.savefig(output_dir + output_filename)
        plt.close()

if __name__ == "__main__":
    main()