# %%
# -*- coding: utf-8 -*-
# EMGを用いて解析を行う

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import global_value as g

###
### 筋電データののみのパスリストを作成
###
def preparation(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" %(g.datafile, ID+1)
    file_list = glob.glob(input_dir)
    
    return file_list

sampling = 2000
task_num = len(g.task)
CI_mean = np.empty(((g.subnum, task_num)))
CI_std = np.empty(((g.subnum, task_num)))
color_map=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]# タスク数に合わせる

for ID in range(g.subnum):
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CI/ancle/PL-PL" %(g.datafile)
    
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    
    CI = pd.DataFrame(np.zeros((g.attempt,len(g.task))), columns=g.task)
    
    file_list = preparation(ID)
    for f in file_list:
        df = pd.read_csv(f)
        taskname = f[(f.find("\\")+1):(f.find("\\")+3)]
        attempt_num = int(f[(f.find("\\")+6)])
        
        # 外れ値を除く
        if ID==1 and taskname=="D2" and attempt_num == 2:
            continue
        if ID==4 and taskname=="D2" and attempt_num == 2:
            continue

        # CI(co-contraction index)の算出
        sum = 0
        sum_PL_R = 0
        sum_PL_L = 0
        for i in range(len(df)):
            # 前脛骨筋：TAが小さいとき，sum_TAにTAを足す
            if df["PL_R"].iloc[i] < df["PL_L"].iloc[i]:
                sum_PL_R = sum_PL_R + df["PL_R"].iloc[i]
            elif df["PL_R"].iloc[i] > df["PL_L"].iloc[i]:
                sum_PL_L = sum_PL_L + df["PL_L"].iloc[i]
            sum = sum + df["PL_R"].iloc[i] + df["PL_L"].iloc[i]
        CI.loc[attempt_num-1, taskname] = (2 * (sum_PL_R + sum_PL_L) / sum) * 100
    
    CI_mean[ID, :] = np.array(CI.mean(axis=0, skipna=True))
    CI_std[ID, :] = np.array(CI.std(axis = 0, skipna=True))
    """
    # 効き足側を0に格納
    if g.domi_leg[ID] == 0:
        CI_mean[ID, :, 0] = np.array(CI_R.mean(axis=0, skipna=True))
        CI_mean[ID, :, 1] = np.array(CI_L.mean(axis=0, skipna=True))
        CI_std[ID, :, 0] = np.array(CI_R.std(axis = 0, skipna=True))
        CI_std[ID, :, 1] = np.array(CI_L.std(axis = 0, skipna=True))        
    else:
        CI_mean[ID, :, 0] = np.array(CI_L.mean(axis=0, skipna=True))
        CI_mean[ID, :, 1] = np.array(CI_R.mean(axis=0, skipna=True))
        CI_std[ID, :, 0] = np.array(CI_L.std(axis = 0, skipna=True))
        CI_std[ID, :, 1] = np.array(CI_R.std(axis = 0, skipna=True))    
    """
    
    ##############################
    #
    # 被験者ごとにプロット
    #
    ##############################
    # グラフをプロット
    # 効き足
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12 
    
    x = np.arange(1, task_num+1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    err = [CI_std[ID, :, 0]]
    ax.bar(x, CI_mean[ID, :, 0], width=0.5, yerr=err, capsize=3, label = g.task, color=color_map)
#    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
#    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0, 120])
    ax.set_ylabel("Co-contraction index of\ndominant foot(%)")
    ax.set_xticks(x)
    ax.set_xticklabels(g.task)
#    plt.show()
    output_filename = "/plot_sub%d_CI_ancle_domi.png" %(ID+1)
    fig.savefig(output_dir + output_filename)
    plt.close()
    
    # グラフをプロット
    # 非利き足
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 12 
    
    x = np.arange(1, task_num+1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    err = [CI_std[ID, :, 1]]
    ax.bar(x, CI_mean[ID, :, 1], width=0.5, yerr=err, capsize=3, label = g.task, color=color_map)
#    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
#    ax.set_xticks(index_num + 0.3)
    ax.set_ylim([0, 120])
    ax.set_ylabel("Co-contraction index of\nnondominant foot(%)")
    ax.set_xticks(x)
    ax.set_xticklabels(g.task)

#    plt.show()
    output_filename = "/plot_sub%d_CI_ancle_nondomi.png" %(ID+1)
    fig.savefig(output_dir + output_filename)
    plt.close()
    """

# データをDataFrame型に戻す
result = pd.DataFrame(CI_mean[[0,2,3,4,5], :], columns=g.task)
std = pd.DataFrame(CI_std[[0,2,3,4,5], :], columns=g.task)

# 新しいファイルに結果を書き込む。
new_file_path = os.path.join(output_dir, "CI_ancle_PL-PL.csv")
result.to_csv(new_file_path)

##############################
#
# 各被験者のデータを横並びに
#
##############################
sublist = []
for i in range(g.subnum):
    if i == 1:
        continue
    sub = "Sub %d" %(i+6)
    sublist.append(sub)

###    
### グラフをプロット　横並び
### 効き足
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sub_num = np.arange(g.subnum-1)
for i in range(task_num):
    slide = i*0.15
    err = [std.iloc[:, i]]
    ax.bar(sub_num+slide, result.iloc[:,i], width=0.12
           , yerr=err, capsize=3, label = g.task[i])
ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
ax.set_xticks(sub_num + 0.3)
#ax.set_ylim([0, 120])
ax.set_ylabel("Co-contraction index of\ndominant foot(%)")
ax.set_xticklabels(sublist)
#plt.show()
output_filename = "/plot_comparison_CI_ancle_domi.png"
fig.savefig(output_dir + output_filename)
plt.close()

"""
###
### グラフをプロット　横並び
### 累積

color_map_2 = ["#17becf","#bcbd22", "#7f7f7f", "#e377c2", "#8c564b"]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cm = plt.get_cmap("tab20")

for i in range(task_num):
    slide = i*0.15
    err_0 = [std_domi.iloc[:, i]]
    err_1 = [std_nondomi.iloc[:, i]]
    ax.bar(sub_num+slide, result_domi.iloc[:,i], width=0.12
           , capsize=3, label = g.task[i], color = color_map[i], edgecolor = "black")
    ax.bar(sub_num+slide, result_nondomi.iloc[:,i], width=0.12
           , capsize=3, label = g.task[i], bottom = result_domi.iloc[:,i], color = cm((i*2)+1), edgecolor = "black", hatch="/")
ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
ax.set_xticks(sub_num + 0.3)
ax.set_ylim([0, 180])
ax.set_ylabel("Co-contraction index of\nfoot cumulative(%)")
ax.set_xticklabels(sublist)
#plt.show()
output_filename = "/plot_comparison_CI_ancle_cumulative.png"
fig.savefig(output_dir + output_filename)
plt.close()
"""