# %%
# -*- coding: utf-8 -*-
# CIの算出を行うコード
# SO-PL
# DBgroup

"""
Created on: 2025-02-05 14:03

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import global_value as g

#task_list = ["NC", "FB", "DBmass", "DBchar", "DW"]
task_list = ["NC", "FB", "DB"]
sampling = 2000
task_num = len(g.task)
color_map = [cm.tab10(i) for i in range(len(task_list))]  # タスク数に合わせる
#new_order = [0, 7, 8, 1, 5, 10]
#new_order = [5, 7, 8, 2, 4, 6]

new_order = [1, 3, 4, 5, 6, 10]# DB〇
#new_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]# 全体


def main():
    # ルートフォルダのパスを指定
    output_dir, output_name = output_preparation()

    CI_mean = np.empty(((g.subnum, task_num, 2)))
    CI_std = np.empty(((g.subnum, task_num, 2)))

    for ID in range(g.subnum):
        sheet_name = "CI_sub%d" % (ID + 1)

        cal_CI(CI_mean, CI_std, ID, output_name, sheet_name)

    # データをDataFrame型に戻す
    mean_domi = pd.DataFrame(CI_mean[:, :3, 0], columns=task_list)
    std_domi = pd.DataFrame(CI_std[:, :3, 0], columns=task_list)

    mean_nondomi = pd.DataFrame(CI_mean[:, :3, 1], columns=task_list)
    std_nondomi = pd.DataFrame(CI_std[:, :3, 1], columns=task_list)

    # 新しいファイルに結果を書き込む。
    new_file_path = os.path.join(output_dir, "CI_ancle_domi.csv")
    mean_domi.to_csv(new_file_path)
    new_file_path = os.path.join(output_dir, "CI_ancle_nondomi.csv")
    mean_nondomi.to_csv(new_file_path)

    ##############################
    #
    # 各被験者のデータを横並びに
    #
    ##############################
    # MVCがうまく取れなかった被験者を除外
    sublist = []
    
    for i in new_order:
        # if i == 1:
        #    continue
        sub = "Sub %d" % (i + 1)
        sublist.append(sub)

    ###
    ### グラフをプロット　横並び
    ### 効き足
    output_filename = "/plot_comparison_CI_ancle_domi.png"
    ylabel = "Co-contraction index of\ndominant foot(%)"
    plt_subject_comparison(
        mean_domi.iloc[new_order], std_domi.iloc[new_order], output_dir, output_filename, sublist, ylabel
    )
    # 非利き足
    output_filename = "/plot_comparison_CI_ancle_nondomi.png"
    ylabel = "Co-contraction index of\nnondominant foot(%)"
    plt_subject_comparison(
        mean_nondomi.iloc[new_order], std_nondomi.iloc[new_order], output_dir, output_filename, sublist, ylabel
    )

    ###
    ### グラフをプロット　横並び
    ### 累積
    output_filename = "/plot_comparison_CI_ancle_cumulative.png"
    ylabel = "Co-contraction index of\ncumulative(%)"
    plt_subject_cumulative(
        mean_domi.iloc[new_order], mean_nondomi.iloc[new_order], output_dir, output_filename, sublist, ylabel
    )

    ##############################
    #
    # 全体の結果を作成
    #
    ##############################
    whole_mean_domi = mean_domi.iloc[new_order].mean(axis=0)
    whole_std_domi = mean_domi.iloc[new_order].std(axis=0)
    whole_mean_nondomi = mean_nondomi.iloc[new_order].mean(axis=0)
    whole_std_nondomi = mean_nondomi.iloc[new_order].std(axis=0)

    # 利き足
    output_filename = "/plot_mean_CI_ancle_domi.png"
    ylabel = "Co-contraction index of\ndominant foot(%)"
    plt_whole(whole_mean_domi, whole_std_domi, output_dir, output_filename, ylabel)

    # 非利き足
    output_filename = "/plot_mean_CI_ancle_nondomi.png"
    ylabel = "Co-contraction index of\nnondominant foot(%)"
    plt_whole(whole_mean_nondomi, whole_std_nondomi, output_dir, output_filename, ylabel)


"""
筋電データののみのパスリストを作成
"""
def preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" % (g.datafile, ID + 1)
    file_list = glob.glob(input_dir)

    return file_list


"""
ファイル名の定義
"""
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CI/ancle/SO-PL/DBgroup" % (g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if os.path.isfile(output_name):
        os.remove(output_name)

    return output_dir, output_name


"""
excelに出力
"""
def excel_output(data, output_name, sheet_name):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name)


"""
CIを計算するメインの関数
"""
def cal_CI(CI_mean, CI_std, ID, output_name, sheet_name):
    CI_R = pd.DataFrame(np.full((g.attempt, len(g.task)), np.nan), columns=g.task)
    CI_L = pd.DataFrame(np.full((g.attempt, len(g.task)), np.nan), columns=g.task)

    file_list = preparation(ID)
    for f in file_list:
        df = pd.read_csv(f)
        taskname = f[(f.find("\\") + 1) : (f.find("\\") + 3)]
        attempt_num = int(f[(f.find("\\") + 6)])

        # 外れ値を除く
        """
        if ID==1 and taskname=="D2" and attempt_num == 2:
            continue
        if ID==4 and taskname=="D2" and attempt_num == 2:
            continue
        """

        # CI(co-contraction index)の算出
        # 右足首
        sum = 0
        sum_SO = 0
        sum_PL = 0
        for i in range(len(df)):
            # 前脛骨筋：SOが小さいとき，sum_SOにSOを足す
            if df["SO_R"].iloc[i] < df["PL_R"].iloc[i]:
                sum_SO = sum_SO + df["SO_R"].iloc[i]
            elif df["SO_R"].iloc[i] > df["PL_R"].iloc[i]:
                sum_PL = sum_PL + df["PL_R"].iloc[i]
            sum = sum + df["SO_R"].iloc[i] + df["PL_R"].iloc[i]
        CI_R.loc[attempt_num - 1, taskname] = (2 * (sum_SO + sum_PL) / sum) * 100

        # 左足首
        sum = 0
        sum_SO = 0
        sum_PL = 0
        for i in range(len(df)):
            # 前脛骨筋：SOが小さいとき，sum_SOにSOを足す
            if df["SO_L"].iloc[i] < df["PL_L"].iloc[i]:
                sum_SO = sum_SO + df["SO_L"].iloc[i]
            elif df["SO_L"].iloc[i] > df["PL_L"].iloc[i]:
                sum_PL = sum_PL + df["PL_L"].iloc[i]
            sum = sum + df["SO_L"].iloc[i] + df["PL_L"].iloc[i]
        CI_L.loc[attempt_num - 1, taskname] = (2 * (sum_SO + sum_PL) / sum) * 100

    CI_R = pd.concat([CI_R, CI_L])
    excel_output(CI_R, output_name, sheet_name)

    # 効き足側を0に格納
    if g.domi_leg[ID] == 0:
        CI_mean[ID, :, 0] = np.array(CI_R.mean(axis=0, skipna=True))
        CI_mean[ID, :, 1] = np.array(CI_L.mean(axis=0, skipna=True))
        CI_std[ID, :, 0] = np.array(CI_R.std(axis=0, skipna=True))
        CI_std[ID, :, 1] = np.array(CI_L.std(axis=0, skipna=True))
    else:
        CI_mean[ID, :, 0] = np.array(CI_L.mean(axis=0, skipna=True))
        CI_mean[ID, :, 1] = np.array(CI_R.mean(axis=0, skipna=True))
        CI_std[ID, :, 0] = np.array(CI_L.std(axis=0, skipna=True))
        CI_std[ID, :, 1] = np.array(CI_R.std(axis=0, skipna=True))


"""
被験者を比較するグラフのプロット
"""
def plt_subject_comparison(mean, std, output_dir, output_filename, sublist, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 20

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 1, 1)
    sub_num = np.arange(len(mean))
    for i in range(len(task_list)):
        slide = i * 0.17
        err = [std.iloc[:, i]]
        ax.bar(
            sub_num + slide,
            mean.iloc[:, i],
            width=0.15,
            yerr=err,
            capsize=3,
            label=task_list[i],
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
    ax.set_xticks(sub_num + 0.3)
    ax.set_ylim([0, 90])
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(sublist)
    # plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()


"""
累積のグラフをプロット
"""


def plt_subject_cumulative(
    mean_domi, mean_nondomi, output_dir, output_filename, sublist, ylabel
):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 20

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.get_cmap("tab20")
    sub_num = np.arange(len(mean_domi))

    for i in range(len(task_list)):
        slide = i * 0.275
        ax.bar(
            sub_num + slide,
            mean_domi.iloc[:, i],
            width=0.25,
            capsize=3,
            label=task_list[i],
            color=color_map[i],
            edgecolor="black",
        )
        ax.bar(
            sub_num + slide,
            mean_nondomi.iloc[:, i],
            width=0.25,
            capsize=3,
            bottom=mean_domi.iloc[:, i],
            color=cm((i * 2) + 1),
            edgecolor="black",
            hatch="/",
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
    ax.set_xticks(sub_num + 0.3)
    ax.set_ylim([0, 180])
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(sublist)
    # plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()


def plt_whole(mean, std, output_dir, output_filename, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12

    x = np.arange(1, len(task_list) + 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    err = [std]
    ax.bar(x, mean, width=0.5, yerr=err, capsize=3, label=task_list, color=color_map)
    #    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    #    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0, 100])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)

    # plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()


if __name__ == "__main__":
    main()
