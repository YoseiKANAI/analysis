# %%
# # -*- coding: utf-8 -*-
"""
Created on: 2024-10-06 14:31

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

task_num = len(g.task)
tasklist = ["NC","FB","DBmass","DBchar", "DW"]

# パスを指定
input_dir = "D:/User/kanai/Data/%s/result_EMG/MDF/*.xlsx" % (g.datafile)
output_dir = "D:/User/kanai/Data/%s/result_EMG/MDF/plot" % (g.datafile)
# 出力先フォルダを作成
os.makedirs(output_dir, exist_ok=True)

def main():
    filename_list = glob.glob(input_dir)
    filename = filename_list[0]
    
    whole_gragh_plot(filename, tasklist)
    sheetname = ["総軌跡長", "SDx", "SDy", "矩形面積", "実効値面積", "標準偏差面積"]
    
    sublist = []
    for i in range(g.subnum):
        sub = "Sub %d" %(i+1)
        sublist.append(sub)
    for index in sheetname:
        index_gragh_plot(filename, index, tasklist, sublist)
     
"""

"""

# 全体グラフをプロットする関数
def whole_gragh_plot(filename, tasklist):
    df = pd.read_excel(filename, sheet_name="正規化後全体")
    mean = df.iloc[[1,2,3, 6,7,8], 1:task_num+1]
    std = df.iloc[[1,2,3, 6,7,8], 8:task_num+9]
    index_num = np.arange(len(mean))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12   

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(task_num):       
        slide = i*0.17
        err = [std.iloc[0][i], std.iloc[1][i], std.iloc[2][i], std.iloc[3][i], std.iloc[4][i], std.iloc[5][i]]
        ax.bar(index_num+slide, mean.iloc[:,i], width=0.17, yerr=err, capsize=3, label = tasklist[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0.0, 1.8])
    ax.set_ylabel("Average values normalized \n By NC values")
    ax.set_xticklabels(["$L_{COP}$", "$\sigma_{x}$", "$\sigma_{y}$", "$S_{rect}$", "$S_{rms}$", "$S_{\sigma}$"])
    #plt.show()
    fig.savefig(output_dir + "/plot_whole.png")

"""

"""

def index_gragh_plot(filename, index, tasklist, sublist):
    df = pd.read_excel(filename, sheet_name=index, header=None)
    mean = df.iloc[44:49, 10:15]
    std = df.iloc[51:56, 10:15]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sub_num = np.arange(g.subnum)
    for i in range(task_num):
        slide = i*0.15
        err = [std.iloc[:, i]]
        ax.bar(sub_num+slide, mean.iloc[:,i], width=0.12
            , yerr=err, capsize=3, label = tasklist[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(sub_num + 0.3)
    ax.set_ylim([0.0, 2.0])
    ax.set_ylabel("Average values normalized \n By NC values")
    ax.set_xticklabels(sublist)
    #plt.show()
    output_filename = "/plot_"+ index +".png"
    fig.savefig(output_dir + output_filename)


if __name__ == "__main__":
    main()
# %%
