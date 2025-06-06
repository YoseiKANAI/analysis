# %%
# # -*- coding: utf-8 -*-
"""
Created on: 2025-01-04 16:40

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
tasklist = ["NC", "FB", "DBmass"]
task_num = len(tasklist)  # NC, FB, DBmassのみ

# パスを指定
input_dir = "D:/User/kanai/Data/%s/result_COP/result/*.xlsx" % (g.datafile)

# 出力先フォルダをどちらかに指定
if mode == 0:
    output_dir = "D:/User/kanai/Data/%s/result_COP/result/boxplot" % (g.datafile)
else:
    output_dir = "D:/User/kanai/Data/%s/result_COP/result/plot_remove_bestworst" % (g.datafile)

# 出力先フォルダを作成
os.makedirs(output_dir, exist_ok=True)


def main():
    filename_list = glob.glob(input_dir)
    filename = filename_list[mode]

    # 標準偏差面積のみ全体グラフを出力（全被験者データをまとめてプロット）
    sheetname = ["標準偏差面積"]
    sublist = [f"Sub {i+1}" for i in range(g.subnum)]
    for index in sheetname:
        df = pd.read_excel(filename, sheet_name=index, header=None)
        # すべての被験者・試行をまとめてタスクごとに集約
        all_data = {task: [] for task in tasklist}
        for i in range(g.subnum):
            data_block = df.iloc[(i*g.attempt)+1 : ((i+1)*g.attempt)+1, 10 : 10+task_num].values
            for t_idx, task in enumerate(tasklist):
                all_data[task].extend(data_block[:, t_idx])
        # プロット
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 24
        plt.figure(figsize=(8, 6))
        data = [all_data[task] for task in tasklist]
        means = [np.mean(all_data[task]) for task in tasklist]
        bplot = plt.boxplot(data, patch_artist=True, labels=tasklist, medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bplot['boxes'], ["tab:blue", "tab:orange", "tab:green"]):
            patch.set_facecolor(color)
        ax = plt.gca()
        for i, mean in enumerate(means):
            ax.plot(i+1, mean, marker='x', color='black', markersize=12, zorder=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0.0, 2.7])
        plt.ylabel("SD Area (normalized)")
        plt.tight_layout()
        plt.savefig(output_dir + "/plot_all_subjects_std_area.png")
        plt.close()

    # --- 以下は個別被験者ごとの従来処理 ---
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
        output_filename = "/plot_" + index + "_normalized.png"
        title = "Average values normalized \n By NC values"
        
        # 5×5×11の空配列を作成
        data = np.zeros((g.attempt, task_num, g.subnum))
        
        for i in range(g.subnum):
            data[:, :, i] = df.iloc[(i*g.attempt)+1 : ((i+1)*g.attempt)+1, 10 : 10+task_num]
        

        index_gragh_plot(
            data,
            output_filename,
            tasklist,
            sublist,
            title,
        )

        output_filename = "/plot_" + index + ".png"
        title = "Average values"
        for i in range(g.subnum):
            data[:, :, i] = df.iloc[(i*g.attempt)+1 : ((i+1)*g.attempt)+1, 10 : 10 +task_num]
        index_gragh_plot(
            data,
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
    mean = df.iloc[[1, 2, 3, 6, 7, 8], 1 : len(tasklist)+1]  # 3条件のみ
    index_num = np.arange(len(mean))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(task_num):
        slide = i * 0.15
        ax.bar(
            index_num + slide,
            mean.iloc[:, i],
            width=0.12,
            capsize=3,
            label=tasklist[i],
        )
    ax.legend(
        loc="upper right",
        fontsize="large",
        ncol=task_num,
        frameon=False,
        handlelength=0.7,
        columnspacing=1,
    )
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0.0, 1.8])
    ax.set_ylabel("Average values normalized \n By NC values")
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
    fig.savefig(output_dir + "/plot_whole.png")


"""
indexごとのグラフを作成する関数
"""


def index_gragh_plot(data, output_filename, tasklist, sublist, title):

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 24

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(1, 1, 1)
    sub_num = np.arange(g.subnum)
    for i in range(task_num):
        df = pd.DataFrame(data[:, i, :], columns=sublist)
        slide = i * 0.15
        ax.boxplot(
            [df[col] for col in df.columns],
            positions = sub_num + slide,
            patch_artist=True,
            widths=0.14,
            boxprops=dict(facecolor="C{}".format(i)),
            medianprops=dict(color="black"),
            labels=sublist if i == 0 else ["" for _ in sublist],
        )
        means = [df[col].mean() for col in df.columns]
        ax.plot(sub_num + slide, means, 'x', color='black', label='Mean' if i == 0 else "")
    ax.legend(
        loc="upper right",
        fontsize="large",
        ncol=task_num,
        frameon=False,
        handlelength=0.7,
        columnspacing=1,
    )
    ax.tick_params(direction="in")
    ax.set_xticks(sub_num + 0.3)
    #    ax.set_ylim([0.0, 2.0])
    ax.set_ylabel(title)
    ax.set_xticklabels(sublist)
    # plt.show()
    fig.savefig(output_dir + output_filename)


if __name__ == "__main__":
    main()
# %%
