# %%
# -*- coding: utf-8 -*-
# サンプルエントロピーをプロットするコード

"""
Created on: 2024-10-24 09:32

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
    summary_mean = np.empty(((g.subnum), g.attempt, 2))
    summary_std = np.empty(((g.subnum), g.attempt, 2))
    output_dir_plot = output_preparation()
    for ID in range(g.subnum):
        if ID == 1:
            continue
        # 出力先フォルダを作成
        filename = input_preparation(ID)
        sheet_name = "SE_sub%d" %(ID+1)
        df = pd.read_excel(filename[0], sheet_name=sheet_name, header=0, index_col=0)
        mean_ind, std_ind = cal_mean_std(df)
        sub_gragh_plot(mean_ind, std_ind, output_dir_plot, ID)
        if g.domi_leg[ID]==0:
            summary_mean[ID, :, 0] = mean_ind["SO_L"]
            summary_std[ID, :, 0] = std_ind["SO_L"]
        else:
            summary_mean[ID, :, 0] = mean_ind["SO_R"]
            summary_std[ID, :, 0] = std_ind["SO_R"]
        if g.domi_arm[ID]==0:
            summary_mean[ID, :, 1] = mean_ind["MF_L"]
            summary_std[ID, :, 1] = std_ind["MF_L"]
        else:
            summary_mean[ID, :, 1] = mean_ind["MF_R"]
            summary_std[ID, :, 1] = std_ind["MF_R"]   
 
    summary_mean = np.delete(summary_mean, 1, axis=0)
    summary_std = np.delete(summary_std, 1, axis=0)
    
    muscle = ["ndomi_SO", "ndomi_MF"]
    summary_gragh_plot(summary_mean, summary_std, output_dir_plot, muscle)

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


def sub_gragh_plot(mean, std, output_dir_plot, ID):
    """
    被験者ごとのグラフをプロットする関数
    """
    index_num = np.arange(g.muscle_num)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12   

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(len(g.task)):       
        slide = i*0.17
        err = [std.iloc[i, :]]
        ax.bar(index_num+slide, mean.iloc[i, :], width=0.17, yerr=err, capsize=3, label = g.task[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=len(g.task), frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0.0, 0.04])
    ax.set_ylabel("Sample Entropy")
    ax.set_xticklabels(g.muscle_columns, rotation = 45)
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
        sub = "Sub %d" %(i+1)
        sublist.append(sub)
    
        sub_num = np.arange(g.subnum-1)

    ###    
    ### グラフをプロット　横並び
    ### 
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
        ax.set_ylim([0, 0.05])
        ax.set_ylabel("Sample Entropy")
        ax.set_xticklabels(sublist)
        #plt.show()
        output_filename = "/plot_%s.png" %(muscle[m])
        fig.savefig(output_dir + output_filename)
        plt.close()


def input_preparation(ID):
    """
    入力パスのリストを作成
    """
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/result_EMG/sample_entropy/*xlsx" %(g.datafile)
    file_list = glob.glob(input_dir)
    
    return file_list

     
def output_preparation():
    """
    ファイル名の定義
    """   
    # ルートフォルダのパスを指定
    output_dir_plot = "D:/User/kanai/Data/%s/result_EMG/sample_entropy/plot" %(g.datafile)
    # 出力先フォルダを作成
    os.makedirs(output_dir_plot, exist_ok=True)
    
    return output_dir_plot
            
if __name__ == "__main__":
    main()
# %%
