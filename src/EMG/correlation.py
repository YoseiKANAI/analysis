# %%
# -*- coding: utf-8 -*-
# TAの相関係数を算出するコード

"""
Created on: 2025-02-05 14:03

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import global_value as g

#task_list = ["NC", "FB", "DBmass", "DBchar", "DW"]
task_list = ["NC", "FB", "DB"]
sampling = 2000
task_num = len(g.task)

def main():
    # ルートフォルダのパスを指定
    output_dir, output_name = output_preparation()

    correlation_data = np.full((g.subnum, g.attempt, task_num), np.nan)

    for ID in range(g.subnum):
        sheet_name = "Correlation_sub%d" % (ID + 1)

        calculate_correlation(correlation_data, ID, output_name, sheet_name)

    # 被験者番号おお3, 5, 7を除外
    exclude_subjects = [2, 4, 6]
    correlation_data = np.delete(correlation_data, exclude_subjects, axis=0)

    # 0次元目と1次元目を結合して次元削減
    correlation_data = correlation_data.reshape(-1, task_num)
    
    # attemptの次元で平均値と標準偏差を求める
    correlation_mean = np.nanmean(correlation_data, axis=0)
    correlation_std = np.nanstd(correlation_data, axis=0)

    # データをDataFrame型に戻す
    mean_correlation = pd.Series(correlation_mean[:3])
    std_correlation = pd.Series(correlation_std[:3])

    # 新しいファイルに結果を書き込む。
    new_file_path = os.path.join(output_dir, "correlation.csv")
    mean_correlation.to_csv(new_file_path)

    # グラフをプロット
    output_filename = "/plot_correlation.svg"
    ylabel = "Correlation coefficient"
    plt_whole(mean_correlation, std_correlation, output_dir, output_filename, ylabel)

def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/correlation/TA" % (g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if os.path.isfile(output_name):
        os.remove(output_name)

    return output_dir, output_name

def calculate_correlation(correlation_data, ID, output_name, sheet_name):
    file_list = preparation(ID)
    for f in file_list:
        df = pd.read_csv(f)
        taskname = f[(f.find("\\") + 1) : (f.find("\\") + 3)]
        attempt_num = int(f[(f.find("\\") + 6)])

        # TA_RとTA_Lの相関係数を算出
        if g.domi_leg[ID] == 0:
            correlation = df["TA_R"].corr(df["TA_L"])
        else:
            correlation = df["TA_L"].corr(df["TA_R"])
        correlation_data[ID, attempt_num - 1, g.task.index(taskname)] = correlation

    # データをExcelに出力
    correlation_df = pd.DataFrame(correlation_data[ID], columns=g.task)
    excel_output(correlation_df, output_name, sheet_name)

def preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" % (g.datafile, ID + 1)
    file_list = glob.glob(input_dir)

    return file_list

def excel_output(data, output_name, sheet_name):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name)

def plt_whole(mean, std, output_dir, output_filename, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    x = np.arange(1, len(task_list) + 1)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    err = [std]
    ax.bar(x, mean, width=0.5, yerr=err, capsize=3, label=task_list)
    ax.tick_params(direction="in")
    ax.set_ylim([-1, 1])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)

    fig.savefig(output_dir + output_filename)
    plt.close()

if __name__ == "__main__":
    main()
