# %%
# # -*- coding: utf-8 -*-
"""
Created on: 2025-03-12 18:13

@author: ShimaLab
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from scipy import stats
from scipy.stats import rankdata


import global_value as g

# 0:通常，1:最高最低を除去
mode = 0
#tasklist = ["NC", "FB"]
tasklist = ["NC", "FB", "DB"]
task_num = len(g.task)
use_task = len(tasklist)


# パスを指定
input_dir = "D:/User/kanai/Data/%s/result_COP/result/*.xlsx" % (g.datafile)

# 出力先フォルダをどちらかに指定
if mode == 0:
    output_dir = "D:/User/kanai/Data/%s/result_COP/result/plot" % (g.datafile)
else:
    output_dir = "D:/User/kanai/Data/%s/result_COP/result/plot_remove_bestworst" % (
        g.datafile
    )

# 出力先フォルダを作成
os.makedirs(output_dir, exist_ok=True)


def main():
    filename_list = glob.glob(input_dir)
    # 0:通常
    # 1:最高最低を除去
    filename = filename_list[mode]

    whole_gragh_plot(filename, tasklist)
    sheetname = ["総軌跡長", "SDx", "SDy", "矩形面積", "実効値面積", "標準偏差面積"]

    sublist = []
    for i in range(g.subnum):
        sub = "Sub %d" % (i + 1)
        sublist.append(sub)
    for index in sheetname:
        comb = math.comb(task_num, 2)
        df = pd.read_excel(filename, sheet_name=index, header=None)
        output_filename = "/plot_" + index + "_normalized.svg"
        title = "Average values normalized \n By NC values"
        mean_normalization = df.iloc[
            g.subnum * (task_num) + comb + 9 : g.subnum * (task_num + 1) + comb + 9,
            task_num + 5 : task_num * 2 + 5,
        ]
        std_normalization = df.iloc[
            g.subnum * (task_num + 1) + comb + 11 : g.subnum * (task_num + 2) + comb + 11,
            task_num + 5 : task_num * 2 + 5,
        ]
        index_gragh_plot(
            mean_normalization.iloc[:, :use_task],
            std_normalization.iloc[:, :use_task],
            output_filename,
            tasklist,
            sublist,
            title,
        )

        output_filename = "/plot_" + index + ".svg"
        title = "Average values"
        mean_non_normalization = df.iloc[
            g.subnum * (task_num) + comb + 9 : g.subnum * (task_num + 1) + comb + 9,
            2 : task_num + 2,
        ]
        std_non_normalization = df.iloc[g.subnum * (task_num + 1) + comb + 11 : g.subnum * (task_num + 2) + comb + 11, 2 : task_num + 2,]
        index_gragh_plot(
            mean_non_normalization,
            std_non_normalization,
            output_filename,
            tasklist,
            sublist,
            title,
        )


"""
全体グラフをプロットする関数
"""
def whole_gragh_plot(filename, tasklist):
    df = pd.read_excel(filename, sheet_name="正規化後全体")
    mean = df.iloc[[1, 2, 3, 6, 7, 8], 1 : task_num + 1]
    std = df.iloc[[1, 2, 3, 6, 7, 8], 8 : task_num + 8]

    #mean = df.iloc[8, 1 : len(tasklist) + 1]
    #std = df.iloc[8, 8 : len(tasklist) + 8]
    index_num = np.arange(1)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(tasklist)):
        slide = i * 0.22
        err = [
            std.iloc[:, i],
            #std[i],
        ]
        ax.bar(
            np.arange(mean.shape[0]) + slide,  # 修正: x軸をインデックス数分に
            mean.iloc[:, i],
            #mean[i],
            width=0.2,  # 幅も調整
            yerr=err,
            capsize=3,
            label=tasklist[i],
        )

    #ax.legend(loc="upper right", fontsize="large", ncol=task_num, frameon=False, handlelength=0.7, columnspacing=1)
    ax.tick_params(direction="in", bottom=False)
    ax.set_xticks(np.arange(mean.shape[0])+0.2)  # 修正: x軸の位置をインデックス数分
    ax.set_ylim([0.0, 1.6])
    #ax.set_ylabel("Average values normalized \n By NC values")
    #ax.set_xticks([])

    ax.set_xticklabels(
        [
            "$L_{COP}$",
            "$\sigma_{x}$",
            "$\sigma_{y}$",
            "$S_{rect}$",
            "$S_{rms}$",
            "$S_{\sigma}$",
        ]
    )
    
    # plt.show()
    fig.savefig(output_dir + "/plot_whole.svg")


"""
indexごとのグラフを作成する関数
"""
def index_gragh_plot(mean, std, output_filename, tasklist, sublist, title):

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 1, 1)
    sub_num = np.arange(len(mean))
    for i in range(len(tasklist)):
        slide = i * 0.29
        err = [std.iloc[:, i]]
        ax.bar(
            sub_num + slide,
            mean.iloc[:, i],
            width=0.28,
            yerr=err,
            capsize=3,
            label=tasklist[i],
        )
    #ax.legend(loc="upper right", fontsize="large", ncol=task_num, frameon=False, handlelength=0.7, columnspacing=1,)
    ax.tick_params(direction="in")
    ax.set_xticks(sub_num + 0.3)
    #ax.set_xticks([])
    #    ax.set_ylim([0.0, 2.0])
    #ax.set_ylabel(title)
    ax.set_xticklabels(sublist, rotation=45, fontsize=12)
    # plt.show()
    fig.savefig(output_dir + output_filename)


if __name__ == "__main__":
    main()
# %%
