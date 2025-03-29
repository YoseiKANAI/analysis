# %%
# -*- coding: utf-8 -*-
# CIの算出を行うコード
# 体幹筋が対称

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import global_value as g

#task_list = ["NC","FB","DBmass","DBchar", "DW"]
task_list = ["NC","FB","DBmass"]
#new_order = [0, 7, 8, 1, 5, 10]
new_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sampling = 2000
task_num = len(g.task)
color_map = [cm.tab10(i) for i in range(len(task_list))]# タスク数に合わせる

def main():        
    # ルートフォルダのパスを指定         
    output_dir, output_name = output_preparation()
    
    CI_mean = np.empty(((g.subnum, task_num, 2)))
    CI_std = np.empty(((g.subnum, task_num, 2)))
    
    for ID in range(g.subnum):    
        sheet_name = "CI_sub%d" %(ID+1)
        
        cal_CI(CI_mean, CI_std, ID, output_name, sheet_name)

    # データをDataFrame型に戻す
    mean_leg = pd.DataFrame(CI_mean[:, :, 0], columns=g.task)
    std_leg = pd.DataFrame(CI_std[:, :, 0], columns=g.task)

    mean_nonleg = pd.DataFrame(CI_mean[:, :, 1], columns=g.task)
    std_nonleg = pd.DataFrame(CI_std[:, :, 1], columns=g.task)

    # 新しいファイルに結果を書き込む。
    new_file_path = os.path.join(output_dir, "CI_MF.csv")
    mean_leg.to_csv(new_file_path)
    new_file_path = os.path.join(output_dir, "CI_IO.csv")
    mean_nonleg.to_csv(new_file_path)

    ##############################
    #
    # 各被験者のデータを横並びに
    #
    ##############################
    sublist = []
    
    for i in new_order:
        #if i == 1:
        #    continue
        sub = "Sub %d" %(i+1)
        sublist.append(sub)

    ###    
    ### グラフをプロット　横並び
    ### 
    # 多裂筋
    output_filename = "/plot_comparison_CI_MF.png"
    ylabel = "Co-contraction index of MF(%)"
    plt_subject_comparison(mean_leg.iloc[new_order], std_leg.iloc[new_order],  output_dir, output_filename, sublist, ylabel)
    # 内腹斜筋
    output_filename = "/plot_comparison_CI_IO.png"
    ylabel = "Co-contraction index of IO(%)"
    plt_subject_comparison(mean_nonleg.iloc[new_order], std_nonleg.iloc[new_order],  output_dir, output_filename, sublist, ylabel)


    ##############################
    #
    # 全体の結果を作成
    #
    ##############################
    whole_mean_leg = mean_leg.mean(axis = 0)
    whole_std_leg = mean_leg.std(axis = 0)
    whole_mean_nonleg = mean_nonleg.mean(axis = 0)
    whole_std_nonleg = mean_nonleg.std(axis = 0)
    
    # 多裂筋
    output_filename = "/plot_mean_CI_MF.png"
    ylabel = "Co-contraction index of MF(%)"
    plt_whole(whole_mean_leg, whole_std_leg, output_dir, output_filename, ylabel)
    
    # 内腹斜筋
    output_filename = "/plot_mean_CI_IO.png"
    ylabel = "Co-contraction index of IO(%)"
    plt_whole(whole_mean_nonleg, whole_std_nonleg, output_dir, output_filename, ylabel)


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
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CI/trunk/RL" %(g.datafile)
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
def cal_CI(CI_mean, CI_std, ID, output_name, sheet_name):
    CI_MF = pd.DataFrame(np.full((g.attempt, len(g.task)), np.nan), columns=g.task)
    CI_IO = pd.DataFrame(np.full((g.attempt, len(g.task)), np.nan), columns=g.task)
    
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
        # 多裂筋
        sum = 0
        sum_R = 0
        sum_L = 0
        for i in range(len(df)):
            R = df["MF_R"].iloc[i]
            L = df["MF_L"].iloc[i]
            # Rが小さいとき，sum_RにRを足す
            if R < L:
                sum_R = sum_R + R
            # Lが小さいとき，sum_LにLを足す
            elif R > L:
                sum_L = sum_L + L
            sum = sum + R + L
            
        CI_MF.loc[attempt_num-1, taskname] = (2 * (sum_R + sum_L) / sum) * 100
        
        # 内腹斜筋
        sum = 0
        sum_R = 0
        sum_L = 0
        for i in range(len(df)):
            R = df["IO_R"].iloc[i]
            L = df["IO_L"].iloc[i]
            # Rが小さいとき，sum_RにRを足す
            if R < L:
                sum_R = sum_R + R
            # Lが小さいとき，sum_LにLを足す
            elif R > L:
                sum_L = sum_L + L
            sum = sum + R + L
            
        CI_IO.loc[attempt_num-1, taskname] = (2 * (sum_R + sum_L) / sum) * 100
    
    CI_MF = pd.concat([CI_MF, CI_IO])
    excel_output(CI_MF, output_name, sheet_name) 
    
    # 多裂筋を0，内腹斜筋を1に格納
    CI_mean[ID, :, 0] = np.array(CI_MF.mean(axis=0, skipna=True))
    CI_mean[ID, :, 1] = np.array(CI_IO.mean(axis=0, skipna=True))
    CI_std[ID, :, 0] = np.array(CI_MF.std(axis=0, skipna=True))
    CI_std[ID, :, 1] = np.array(CI_IO.std(axis=0, skipna=True))        


"""
被験者を比較するグラフのプロット
"""
def plt_subject_comparison(mean, std,  output_dir, output_filename, sublist, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 20

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1,1,1)
    sub_num = np.arange(len(mean))
    for i in range(len(task_list)):
        slide = i*0.15
        err = [std.iloc[:, i]]
        ax.bar(sub_num+slide, mean.iloc[:,i], width=0.15
            , yerr=err, capsize=3, label = task_list[i])
    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(sub_num + 0.3)
    ax.set_ylim([0, 100])
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
    plt.rcParams["font.size"] = 12   

    x = np.arange(1, task_num+1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    err = [std]
    ax.bar(x, mean, width=0.5, yerr=err, capsize=3, label = task_list, color=color_map)
    #    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    #    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0, 100])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)

    #plt.show()
    fig.savefig(output_dir + output_filename)
    plt.close()

if __name__ == "__main__":
    main()