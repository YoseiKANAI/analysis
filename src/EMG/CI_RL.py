# %%
# -*- coding: utf-8 -*-
# CIの算出を行うコード
# SO

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

#task_list = ["NC","FB","DBmass","DBchar", "DW"]
task_list = ["NC","FB","DB"]
sampling = 2000
task_num = len(g.task)
color_map = [cm.tab10(i) for i in range(len(task_list))]# タスク数に合わせる
#new_order = [0, 7, 8, 1, 5, 10]
#new_order = [0, 1, 3, 5, 7, 8, 9, 10]
#exclude_subjects = [2, 4, 6]

new_order = [1, 3, 4, 5, 6, 10]
exclude_subjects = [0, 2, 7, 8, 9]
def main():        
    # ルートフォルダのパスを指定         
    output_dir, output_name = output_preparation()
    
    CI = np.full((g.subnum, g.attempt, task_num), np.nan)
    
    normalized_data = []

    for ID in range(g.subnum):    
        sheet_name = "CI_sub%d" %(ID+1)
        
        cal_CI(CI, ID, output_name, sheet_name)
        
    # 被験者番号3, 5, 7を除外
    
    CI = np.delete(CI, exclude_subjects, axis=0)

    # 0次元目と1次元目を連結して次元削減
    CI = CI.reshape(-1, task_num)

    # attemptの次元で平均値と標準偏差を求める
    CI_mean = np.nanmean(CI, axis=0)
    CI_std = np.nanstd(CI, axis=0)

    # データをDataFrame型に戻す
    mean = pd.Series(CI_mean[:3])
    std = pd.Series(CI_std[:3])

    # グラフをプロット
    output_filename = "/plot_normalized_CI_RL_SO.svg"
    ylabel = "Normalized Co-contraction index(%)"
    plt_whole(mean, std, output_dir, output_filename, ylabel)
    
    """
    # データをDataFrame型に戻す
    mean_leg = pd.DataFrame(CI_mean[:, :3], columns=task_list)
    std_leg = pd.DataFrame(CI_std[:, :3], columns=task_list)

    # 新しいファイルに結果を書き込む。
    new_file_path = os.path.join(output_dir, "CI_trunk_leg.csv")
    mean_leg.to_csv(new_file_path)
    new_file_path = os.path.join(output_dir, "CI_trunk_nonleg.csv")
    mean_nonleg.to_csv(new_file_path)

    ##############################
    #
    # 各被験者のデータを横並びに
    #
    ##############################
    # MVCがうまく取れなかった被験者を除外
    sublist = []
    for i in new_order:
        #if i == 1:
        #    continue
        sub = "Sub %d" %(i+1)
        sublist.append(sub)

    ###    
    ### グラフをプロット　横並び
    ### 効き足
    output_filename = "/plot_comparison_CI_trunk_leg.svg"
    ylabel = "Co-contraction index of\ndominant foot(%)"
    plt_subject_comparison(mean_leg.iloc[new_order], std_leg.iloc[new_order],  output_dir, output_filename, sublist, ylabel)
    # 非利き足
    output_filename = "/plot_comparison_CI_trunk_nonleg.svg"
    ylabel = "Co-contraction index of\nnondominant leg(%)"
    plt_subject_comparison(mean_nonleg.iloc[new_order], std_nonleg.iloc[new_order],  output_dir, output_filename, sublist, ylabel)

    ###
    ### グラフをプロット　横並び
    ### 累積
    output_filename = "/plot_comparison_CI_trunk_cumulative.svg"
    ylabel = "Co-contraction index of\ncumulative(%)"
    plt_subject_cumulative(mean_leg.iloc[new_order], mean_nonleg.iloc[new_order],  output_dir, output_filename, sublist, ylabel)


    ##############################
    #
    # 全体の結果を作成
    #
    ##############################
    whole_mean_leg = mean_leg.mean(axis = 0)
    whole_std_leg = mean_leg.std(axis = 0)
    whole_mean_nonleg = mean_nonleg.mean(axis = 0)
    whole_std_nonleg = mean_nonleg.std(axis = 0)
    
    # 利き足
    output_filename = "/plot_mean_CI_trunk_leg.svg"
    ylabel = "Co-contraction index of\ndominant leg(%)"
    plt_whole(whole_mean_leg, whole_std_leg, output_dir, output_filename, ylabel)
    
    # 非利き足
    output_filename = "/plot_mean_CI_trunk_nonleg.svg"
    ylabel = "Co-contraction index of\nnondominant leg(%)"
    plt_whole(whole_mean_nonleg, whole_std_nonleg, output_dir, output_filename, ylabel)
    """


"""
筋電データののみのパスリストを作成
"""
def preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" %(g.datafile, ID+1)
    file_list = glob.glob(input_dir)
    
    return file_list

"""
ファイル名の定義
"""   
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CI/RL/SO/DBgroup" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_dir, output_name
    
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
CIを計算するメインの関数
"""
def cal_CI(CI, ID, output_name, sheet_name):
    result = pd.DataFrame(np.full((g.attempt, len(g.task)), np.nan), columns=g.task)
    
    file_list = preparation(ID)
    for f in file_list:
        df = pd.read_csv(f)
        taskname = f[(f.find("\\")+1):(f.find("\\")+3)]
        attempt_num = int(f[(f.find("\\")+6)])
        
        # 外れ値を除く
        """
        if ID==1 and taskname=="D2" and attempt_num == 2:
            continue
        if ID==4 and taskname=="D2" and attempt_num == 2:
            continue
        """

        # CI(co-contraction index)の算出
        # 右側腹斜筋と左多裂筋
        sum = 0
        sum_R = 0
        sum_L = 0
        for i in range(len(df)):
            R = df["SO_R"].iloc[i]
            L = df["SO_L"].iloc[i]
            # 多裂筋：Rが小さいとき，sum_RにRを足す
            if R < L:
                sum_R = sum_R + R
            # 腹斜筋：Lが小さいとき，sum_LにLを足す
            elif R > L:
                sum_L = sum_L + L
            sum = sum + R + L
            
        result.loc[attempt_num-1, taskname] = (2 * (sum_R + sum_L) / sum) * 100
        
    
    NC_mean = result.iloc[:, 0].mean(axis=0, skipna=True)
    
    CI_norm = result / NC_mean

    CI[ID, :, :] = np.array(CI_norm)

    excel_output(CI_norm , output_name, sheet_name) 

"""
被験者を比較するグラフのプロット
"""
def plt_subject_comparison(mean, std,  output_dir, output_filename, sublist, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1,1,1)
    sub_num = np.arange(len(sublist))
    for i in range(len(task_list)):
        slide = i*0.275
        err = [std.iloc[:, i]]
        ax.bar(sub_num+slide, mean.iloc[:,i], width=0.25)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(sublist)
    #plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()
    
"""
累積のグラフをプロット
"""
def plt_subject_cumulative(mean_leg, mean_nonleg,  output_dir, output_filename, sublist, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 20

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1,1,1)
    cm = plt.get_cmap("tab20")
    sub_num = np.arange(len(sublist))
    
    for i in range(len(task_list)):
        slide = i*0.15
        ax.bar(sub_num+slide, mean_leg.iloc[:,i], width=0.15
            , capsize=3, label = task_list[i], color = color_map[i], edgecolor = "black")
        ax.bar(sub_num+slide, mean_nonleg.iloc[:,i], width=0.15
            , capsize=3, bottom = mean_leg.iloc[:,i], color = cm((i*2)+1), edgecolor = "black", hatch="/")
    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(sub_num + 0.3)
    ax.set_ylim([0, 200])
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(sublist)
    #plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()


"""
全体グラフをプロットする関数
"""
def plt_whole(mean,std , output_dir, output_filename, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18   

    x = np.arange(1, len(task_list)+1)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1,1,1)
    err = [std]
    ax.bar(x, mean, width=0.5, yerr=err, capsize=3, label = task_list, color=color_map)
    #    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    #    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0, 1.4])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)

    #plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()

if __name__ == "__main__":
    main()