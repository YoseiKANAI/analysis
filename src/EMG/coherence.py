# %%
# -*- coding: utf-8 -*-
# 筋間コヒーレンスの算出
# TA-PL
#

"""
Created on: 2024-09-24 17:28

@author: ShimaLab
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import coherence

import global_value as g

sampling = 2000
task_num = len(g.task)
freq_list = [
    "2-8Hz",
    "8-16Hz",
    "20-40Hz",
    "40-60Hz",
]

color_map = [cm.tab10(i) for i in range(len(g.task))]  # タスク数に合わせる
#task_list = ["NC", "FB", "DBmass", "DBchar", "DW"]
task_list = ["No Contact", "Floating Balloon", "Dropping Balloon"]

# グループの定義
set_group_1 = {3, 4, 7, 10}# 両方×
set_group_2 = {1, 8, 9}# DBのみ×
set_group_3 = {6, 11}# 両方〇

"""
set_group_1 = {1, 2, 4, 6, 8, 9, 10, 11}#　FB〇
set_group_2 = {3}# DBのみ×
set_group_3 = {5}# 両方〇
"""
# 情報処理用
""""
set_group_1 = {1,2, 4, 6, 8, 9, 10, 11}#　FB〇
set_group_2 = {2, 4, 5, 6, 7, 11}# DB〇
set_group_3 = {2, 4, 6, 11}# 両方〇
"""

"""
main関数
"""
def main():
    # 群ごとの生データ
    result_summary_group1 = [pd.DataFrame() for _ in range(5)]
    result_summary_group2 = [pd.DataFrame() for _ in range(5)]
    result_summary_group3 = [pd.DataFrame() for _ in range(5)]
    # 全体の生データ
    result_summary_whole = [pd.DataFrame() for _ in range(5)]
    # 被験者ごとに正規化した積分値 
    integral_sub_nomalized = np.full((g.subnum, g.attempt, len(freq_list), len(task_list)) , np.nan)
    
    output_name, output_dir_plot = output_preparation()
    for ID in range(g.subnum):
        filename_list = input_preparation(ID)
        output_dir_indi = output_preparation_indi(ID)
        sheet_name = "sub%d" % (ID + 1)

        # コヒーレンスを算出
        coherence = create_coherence_data(filename_list, output_dir_indi, ID, result_summary_group1, result_summary_group2, result_summary_group3, result_summary_whole)
        # 平均と分散を導出
        #mean_indi, std_indi = cal_mean_std(coherence)
        mean_indi = pd.DataFrame()
        std_indi = pd.DataFrame()
        
        # 群ごとの平均と分散を算出
        for df in coherence:
        # タスクごとの平均と分散を算出
            mean_indi = pd.concat([mean_indi, df.mean(axis=1, skipna=True)], axis=1)
            std_indi = pd.concat([std_indi, df.std(axis=1, skipna=True, ddof=1)], axis=1)
        mean_indi.columns = g.task
        std_indi.columns = g.task
        # 生データをプロット
        raw_graph_plot(mean_indi.iloc[:, :len(task_list)], output_dir_plot, sheet_name)

        integral_indi = np.full((coherence[0].shape[1], len(freq_list), task_num), np.nan)
        # 周波数帯ごとに積分
        #coherence = integral_coherence(mean_indi)
        # グループごとの積分値を算出
        for t in range(task_num):
            for i in range(coherence[t].shape[1]):
                integral_indi[i, :, t] = integral_coherence(pd.DataFrame(coherence[t].iloc[:, i]))
        # 積分したデータをエクセル用にまとめる
        #coherence = pd.concat([mean_indi, pd.DataFrame(columns=[""]), i_coherence], axis=1)
        #excel_output(coherence, output_name, sheet_name)
        
        # 被験者ごとの正規分値を算出
        mean_integral_indi = np.nanmean(integral_indi, axis=0)
        std_integral_indi = np.nanstd(integral_indi, axis=0, ddof=1)
        integral_gragh_plot(mean_integral_indi[:, :len(task_list)], std_integral_indi[:, :len(task_list)], output_dir_plot, ID)
        
        #
        integral_sub_nomalized[ID, :, :, :] =  integral_indi[:, :, :len(task_list)] / mean_integral_indi[:, 0][:, np.newaxis]
    
    ###
    ### 群ごとの平均と分散を算出
    ###
    mean_group1 = pd.DataFrame()
    std_group1 = pd.DataFrame()
    mean_group2 = pd.DataFrame()
    std_group2 = pd.DataFrame()
    mean_group3 = pd.DataFrame()
    std_group3 = pd.DataFrame()
    mean_whole = pd.DataFrame()
    std_whole = pd.DataFrame()
    
    # 群ごとのデータをまとめるデータフレーム
    result_summary_group1_all = pd.DataFrame()
    result_summary_group2_all = pd.DataFrame()
    result_summary_group3_all = pd.DataFrame()
    result_summary_whole_all = pd.DataFrame()
    
    # 群ごとの平均と分散を算出
    # グループ1
    for df in result_summary_group1:
        # タスクごとの平均と分散を算出
        mean_group1 = pd.concat([mean_group1, df.mean(axis=1, skipna=True)], axis=1)
        std_group1 = pd.concat([std_group1, df.std(axis=1, skipna=True, ddof=1)], axis=1)
        # 全データの平均と分散
        result_summary_group1_all = pd.concat([result_summary_group1_all, df], axis=1)
        mean_group1_all = result_summary_group1_all.mean(axis=1, skipna=True)
        std_group1_all = result_summary_group1_all.std(axis=1, skipna=True, ddof=1)
    
    # グループ2
    for df in result_summary_group2:
        # タスクごとの平均と分散を算出
        mean_group2 = pd.concat([mean_group2, df.mean(axis=1, skipna=True)], axis=1)
        std_group2 = pd.concat([std_group2, df.std(axis=1, skipna=True, ddof=1)], axis=1)
        # 全データの平均と分散
        result_summary_group2_all = pd.concat([result_summary_group2_all, df], axis=1)
        mean_group2_all = result_summary_group2_all.mean(axis=1, skipna=True)
        std_group2_all = result_summary_group2_all.std(axis=1, skipna=True, ddof=1)
    
    # グループ3
    for df in result_summary_group3:
        # タスクごとの平均と分散を算出
        mean_group3 = pd.concat([mean_group3, df.mean(axis=1, skipna=True)], axis=1)
        std_group3 = pd.concat([std_group3, df.std(axis=1, skipna=True, ddof=1)], axis=1)
        # 全データの平均と分散
        result_summary_group3_all = pd.concat([result_summary_group3_all, df], axis=1)
        mean_group3_all = result_summary_group3_all.mean(axis=1, skipna=True)
        std_group3_all = result_summary_group3_all.std(axis=1, skipna=True, ddof=1)
    
    # 全体
    for df in result_summary_whole:
        # タスクごとの平均と分散を算出
        mean_whole = pd.concat([mean_whole, df.mean(axis=1, skipna=True)], axis=1)
        std_whole = pd.concat([std_whole, df.std(axis=1, skipna=True, ddof=1)], axis=1)
    
    mean_group1.columns = g.task
    std_group1.columns = g.task
    mean_group2.columns = g.task
    std_group2.columns = g.task
    mean_group3.columns = g.task
    std_group3.columns = g.task
    mean_whole.columns = g.task
    std_whole.columns = g.task
    
    # グループごとの波形をプロット
    raw_graph_plot(mean_group1, output_dir_plot, "mean_group1")
    raw_graph_plot(std_group1, output_dir_plot, "std_group1")
    raw_graph_plot(mean_group2, output_dir_plot, "mean_group2")
    raw_graph_plot(std_group2, output_dir_plot, "std_group2")
    raw_graph_plot(mean_group3, output_dir_plot, "mean_group3")
    raw_graph_plot(std_group3, output_dir_plot, "std_group3")
    raw_graph_plot(mean_whole, output_dir_plot, "mean_whole")
    raw_graph_plot(std_whole, output_dir_plot, "std_whole")
    
    
    ###
    ### タスク関係なしで算出
    ###
    mean_all = pd.concat([mean_group1_all, mean_group2_all, mean_group3_all], axis=1)
    std_all = pd.concat([std_group1_all, std_group2_all, std_group3_all], axis=1)

    mean_all.columns = ["group1", "group2", "group3"]
    std_all.columns = ["group1", "group2", "group3"]
    #　タスク関係なしで算出した波形をプロット
    raw_graph_plot(mean_all, output_dir_plot, "Mean group1 vs group2 vs group3")
    raw_graph_plot(std_all, output_dir_plot, "Std group1 vs group2 vs group3")


    ###
    ### 各グループで積分値を算出
    ###
    integral_group1 = np.full((result_summary_group1[0].shape[1], len(freq_list), task_num), np.nan)
    integral_group2 = np.full((result_summary_group2[0].shape[1], len(freq_list), task_num), np.nan)
    integral_group3 = np.full((result_summary_group2[0].shape[1], len(freq_list), task_num), np.nan)
    # 全体で積分値を算出
    integral_whole = np.full((result_summary_whole[0].shape[1], len(freq_list), task_num), np.nan)
    
    # グループごとの積分値を算出
    for t in range(task_num):
        for i in range(result_summary_group1[t].shape[1]):
            integral_group1[i, :, t] = integral_coherence(pd.DataFrame(result_summary_group1[t].iloc[:, i]))
        for i in range(result_summary_group2[t].shape[1]):
            integral_group2[i, :, t] = integral_coherence(pd.DataFrame(result_summary_group2[t].iloc[:, i]))
        for i in range(result_summary_group3[t].shape[1]):
            integral_group3[i, :, t] = integral_coherence(pd.DataFrame(result_summary_group3[t].iloc[:, i]))
        for i in range(result_summary_whole[t].shape[1]):
            integral_whole[i, :, t] = integral_coherence(pd.DataFrame(result_summary_whole[t].iloc[:, i]))

    # 各データをプロット
    # グループ1
    mean_integral_group1 = np.nanmean(integral_group1, axis=0)
    std_integral_group1 = np.nanstd(integral_group1, axis=0, ddof=1)
    # グループ2
    mean_integral_group2 = np.nanmean(integral_group2, axis=0)
    std_integral_group2 = np.nanstd(integral_group2, axis=0, ddof=1)
    # グループ3
    mean_integral_group3 = np.nanmean(integral_group3, axis=0)
    std_integral_group3 = np.nanstd(integral_group3, axis=0, ddof=1)
    
    # グループ間の比較をプロット
    plot_group_comparison(mean_integral_group1, std_integral_group1, mean_integral_group2, std_integral_group2, mean_integral_group3, std_integral_group3, output_dir_plot)
    
    # 全体の平均と分散を算出
    mean_integral_whole = np.nanmean(integral_whole, axis=0)
    std_integral_whole = np.nanstd(integral_whole, axis=0, ddof=1)
    integral_gragh_plot(mean_integral_whole[:, :3], std_integral_whole, output_dir_plot, g.subnum)
    
    # 被験者ごと正規化値で算出
    # sub_normarizedの平均を算出
    # integral_sub_nomalizedの0次元目が被験者，1次元目が試行，2次元目が周波数帯，3次元目がタスク
    # 配列の形状を[被験者×試行，周波数帯，タスク]に変更
    integral_sub_nomalized_whole = integral_sub_nomalized.reshape(g.subnum * g.attempt, len(freq_list), len(task_list))
    
    mean_integral_sub_nomalized_whole = np.nanmean(integral_sub_nomalized_whole, axis=0)
    std_integral_sub_nomalized_whole = np.nanstd(integral_sub_nomalized_whole, axis=0, ddof=1)
    integral_gragh_plot(mean_integral_sub_nomalized_whole, std_integral_sub_nomalized_whole, output_dir_plot, g.subnum)
    
     ###
    ### 統計用のデータを格納
    ###
    # 全体
    to_excel_certification_data(integral_whole, output_name, "whole")
    # グループ1
    to_excel_certification_data(integral_group1, output_name, "group1")
    # グループ2
    to_excel_certification_data(integral_group2, output_name, "group2")
    # グループ3
    to_excel_certification_data(integral_group3, output_name, "group3")
    # 被験者ごとの正規化データ
    to_excel_certification_data(integral_sub_nomalized_whole, output_name, "sub_nomalized")
    
    # group1に関して
    list_group_1 = [x - 1 for x in list(set_group_1)]
    integral_sub_nomalized_group1 = integral_sub_nomalized[list_group_1, :, :, :].reshape(len(list_group_1) * g.attempt, len(freq_list), len(task_list))
    
    mean_integral_sub_nomalized_group1 = np.nanmean(integral_sub_nomalized_group1, axis=0)
    std_integral_sub_nomalized_group1 = np.nanstd(integral_sub_nomalized_group1, axis=0, ddof=1)
    integral_gragh_plot(mean_integral_sub_nomalized_group1, std_integral_sub_nomalized_group1, output_dir_plot, 15)    
    



"""
MeanEMGAmplitudeを計算する関数
"""
def create_coherence_data(filename_list, output_dir_indi, ID, result_summary_group1, result_summary_group2, result_summary_group3, result_summary_whole):
    # 格納するDataFrameを作成
    result = [pd.DataFrame() for _ in range(5)]
    # リストの初期化
    task_list = []
    output_path_plot = output_dir_indi + "/plot"
    os.makedirs(output_path_plot, exist_ok=True)
    # リストの順に呼び出し
    for t in g.task:
        task_list = [s for s in filename_list if t in s]
        # ファイルごとに計算
        for f in task_list:
            df, columns_list = csv_reader(f)

            attempt_num = int(f[(f.find("\\") + 6)])
            index_name = t + "_%s" % (attempt_num)
            output_file_name_indi = index_name + ".csv"
            output_path_indi = os.path.join(output_dir_indi, output_file_name_indi)
        
            """
            if t=="D2" and attempt_num == 2 and ID ==1:
                result = pd.concat([result, pd.DataFrame()])
                continue
            if t=="D2" and attempt_num == 2 and ID ==4:
                result = pd.concat([result, pd.DataFrame()])
                continue
            """
            index = g.task.index(t)
            coherence = cal_coherence(df, ID, index_name)
            
            raw_graph_plot(coherence, output_path_plot, index_name)
            result[index] = pd.concat([result[index], coherence], axis=1)
            #coherence.to_csv(output_path_indi)
            
            if ID+1 in set_group_1:# 両方×
                result_summary_group1[index] = pd.concat([result_summary_group1[index], coherence], axis=1)
            if ID+1 in set_group_2:# DBのみ×
                result_summary_group2[index] = pd.concat([result_summary_group2[index], coherence], axis=1)
            if ID+1 in set_group_3:# 両方〇
                result_summary_group3[index] = pd.concat([result_summary_group3[index], coherence], axis=1)
        

            result_summary_whole[index] = pd.concat([result_summary_whole[index], coherence], axis=1)
            a=0
            
            #result = pd.concat([result, pd.DataFrame(coherence)], axis=1)
    return result


"""
coherenceの計算を行う関数
"""
def cal_coherence(df, ID, index_name):
    if g.domi_leg[ID] == 0:
        EMG_1 = df["TA_R"].ffill().bfill()
        EMG_2 = df["PL_R"].ffill().bfill()
    elif g.domi_leg[ID] == 1:
        EMG_1 = df["TA_L"].ffill().bfill()
        EMG_2 = df["PL_L"].ffill().bfill()
    """
    EMG_1 = df["PL_R"]
    EMG_2 = df["PL_L"]
    """
    freq, coherence_value = coherence(EMG_1, EMG_2, fs=sampling, nperseg=2048)
    result_indi = pd.DataFrame(coherence_value[:105], index=freq[:105], columns = [index_name])
    return result_indi


"""
被験者ごとの平均と分散を算出し，格納
"""
def cal_mean_std(result):
    mean = pd.DataFrame(columns=g.task)
    std = pd.DataFrame(columns=g.task)
    # タスクごとの平均と分散を算出
    for i in range(task_num):
        # coherenceからタスクごとのデータを取り出し
        df = result.iloc[:, i * 5 : (i + 1) * 5]

        # 平均をcoherenceに格納
        mean[g.task[i]] = df.mean(axis=1, skipna=True)
        std[g.task[i]] = df.std(axis=1, skipna=True, ddof=1)
    return mean, std


"""
コヒーレンスの周波数帯ごとの積分値を算出
"""
def integral_coherence(coherence):
    result = pd.DataFrame()
    """
    # デルタ帯　1~4Hz
    delta = (coherence.iloc[1:5]).sum(axis=0)
    # シータ帯　4~8Hz
    theta = (coherence.iloc[5:9]).sum(axis=0)
    # アルファ帯　8~14Hz
    alpha = (coherence.iloc[9:15]).sum(axis=0)
    # ベータ帯　14~30Hz
    beta = (coherence.iloc[15:31]).sum(axis=0)
    # ガンマ帯　30~60Hz
    gamma = (coherence.iloc[31:62]).sum(axis=0)
    """
    
    # デルタ帯　2~8Hz
    delta = (coherence.iloc[3:10]).sum(axis=0)
    # シータ帯　8~16Hz
    theta = (coherence.iloc[10:18]).sum(axis=0)
    # アルファ帯　20~40Hz
    alpha = (coherence.iloc[21:44]).sum(axis=0)
    # ベータ帯　40~60Hz
    beta = (coherence.iloc[44:66]).sum(axis=0)
    """
    # デルタ帯　0~6Hz
    delta = (coherence.iloc[:8]).sum(axis=0)
    # シータ帯　8~14Hz
    theta = (coherence.iloc[9:16]).sum(axis=0)
    # アルファ帯　15~30Hz
    alpha = (coherence.iloc[17:33]).sum(axis=0)
    # ベータ帯　38~42Hz
    beta = (coherence.iloc[40:43]).sum(axis=0)
    """
    result = pd.concat([delta, theta, alpha, beta], axis=1)
    #result.index = freq_list

    return result


"""
入力パスのリストを作成
"""
def input_preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/*a_2.csv" % (g.datafile, ID + 1)
    file_list = glob.glob(input_dir)

    return file_list


"""
ファイル名の定義
"""
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/coherence_IPSJ/TA-PL" % (g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_EMG/coherence_IPSJ/TA-PL/plot" % (g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if os.path.isfile(output_name):
        os.remove(output_name)

    return output_name, output_dir_plot


"""
個別ファイルの格納先指定
"""
def output_preparation_indi(ID):
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/coherence/TA-PL" % (g.datafile, ID + 1)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


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
    for i in range(task_num):  # タスク数
        for j in range(g.attempt):  # 試行数
            for k in range(g.muscle_num):  # 筋電数
                sum[i, (ID * g.attempt) + j, k] = result.iloc[
                    (i * g.attempt) + j + 43, k
                ]
    return sum


"""
生データのグラフをプロットする関数
"""
def raw_graph_plot(coherence, output_dir_plot, sheet_name):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    #ax.set_xlabel("Freqency [Hz]")
    #ax.set_ylabel("Coherence 0-1")
    ax.set_xlim([0, 60])  # x方向の描画範囲を指定
    ax.set_ylim([0, 0.25])  # y方向の描画範囲を指定

    for t in coherence.iloc[:, :3]:
        ax.plot(coherence[t], label=t)
    #ax.legend()

    output_plot_raw = output_dir_plot + "/" + sheet_name + ".svg"
    plt.savefig(output_plot_raw)  # 画像の保存
    plt.close(fig)
    plt.clf()


"""
被験者ごとのグラフをプロットする関数
"""
def integral_gragh_plot(mean, std, output_dir_plot, ID):
    index_num = np.arange(len(freq_list))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    for t in range(len(task_list)):
        slide = t * 0.275
        ax.bar(
            index_num + slide, mean[:, t], yerr=std[:, t], width=0.25, capsize=3, label=task_list[t]
        )
    ax.legend(
        loc="upper center",
        fontsize="large",
        ncol=task_num,
        frameon=False,
        handlelength=0.7,
        columnspacing=1,
    )
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0.0, 2.2])
    #ax.set_ylabel("Coherence (0-1)")
    ax.set_xticklabels(freq_list)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.legend(loc='upper center', bbox_to_anchor=(1, 1))
    # plt.show()
    plot_name = output_dir_plot + "/sub%d_integral.svg" % (ID + 1)
    fig.savefig(plot_name)
    plt.close(fig)
    plt.clf()


"""
グループ間の比較をプロット
"""
def plot_group_comparison(mean_group1, std_group1, mean_group2, std_group2, mean_group3, std_group3, output_dir):
    index = np.arange(len(freq_list))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    
    for t in range(len(task_list)):
        slide = t * 0.275
        # err = [result.iloc[:, i]]
        ax.bar(index + slide - 4.5, mean_group1[:, t], yerr=std_group1[:, t], width=0.25, capsize=3, label=task_list[t], color=color_map[t])
        ax.bar(index + slide, mean_group2[:, t], yerr=std_group2[:, t], width=0.25, capsize=3, color=color_map[t])
        ax.bar(index + slide + 4.5, mean_group3[:, t], yerr=std_group3[:, t], width=0.25, capsize=3, color=color_map[t])

    # x軸方向のラベルとメモリをなくす
    ax.set_xticklabels([])
    ax.set_xticks([])
    # 上と右側の線を消す
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'group_comparison.svg')
    plt.savefig(output_path)
    plt.close(fig)
    plt.clf()


"""
excelに出力
"""
def excel_output(data, output_name, sheet_name):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)


"""
統計用のデータを格納する関数
"""
def to_excel_certification_data(data, output_name, sheet_name):
    integral_flat = pd.DataFrame()
    for i in range(len(task_list)):
        integral_flat = pd.concat([integral_flat, pd.DataFrame(data[:, :, i])])
        #integral_flat.columns = freq_list
    excel_output(integral_flat, output_name, sheet_name)

if __name__ == "__main__":
    main()
# %%
