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

def output_preparation():
    """
    ファイル名の定義
    """   
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CI/ancle/PL-TA" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_dir, output_name

def excel_output(data, output_name, sheet_name):
    """
    excelに出力
    """
    if (os.path.isfile(output_name)):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name = sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name = sheet_name)
        
# ルートフォルダのパスを指定         
output_dir, output_name = output_preparation()
sampling = 2000
task_num = len(g.task)
result_np_summary = np.empty(((g.subnum, task_num, 2)))
CI_std = np.empty(((g.subnum, task_num, 2)))
color_map=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]# タスク数に合わせる

for ID in range(g.subnum):    
    sheet_name = "SE_sub%d" %(ID+1)
    CI_R = pd.DataFrame(np.empty((g.attempt,len(g.task))), columns=g.task)
    CI_L = pd.DataFrame(np.empty((g.attempt,len(g.task))), columns=g.task)
    
    file_list = preparation(ID)
    for f in file_list:
        df = pd.read_csv(f)
        taskname = f[(f.find("\\")+1):(f.find("\\")+3)]
        attempt_num = int(f[(f.find("\\")+6)])
        
        """
        # 外れ値を除く
        if ID==1 and taskname=="D2" and attempt_num == 2:
            continue
        if ID==4 and taskname=="D2" and attempt_num == 2:
            continue
        """
        
        # CI(co-contraction index)の算出
        # 右足首
        sum = 0
        sum_PL = 0
        sum_TA = 0
        for i in range(len(df)):
            # 前脛骨筋：PLが小さいとき，sum_PLにPLを足す
            if df["PL_R"].iloc[i] < df["TA_R"].iloc[i]:
                sum_PL = sum_PL + df["PL_R"].iloc[i]
            elif df["PL_R"].iloc[i] > df["TA_R"].iloc[i]:
                sum_TA = sum_TA + df["TA_R"].iloc[i]
            sum = sum + df["PL_R"].iloc[i] + df["TA_R"].iloc[i]
        CI_R.loc[attempt_num-1, taskname] = (2 * (sum_PL + sum_TA) / sum) * 100
        
        # 左足首
        sum = 0
        sum_PL = 0
        sum_TA = 0
        for i in range(len(df)):
            # 前脛骨筋：PLが小さいとき，sum_PLにPLを足す
            if df["PL_L"].iloc[i] < df["TA_L"].iloc[i]:
                sum_PL = sum_PL + df["PL_L"].iloc[i]
            elif df["PL_L"].iloc[i] > df["TA_L"].iloc[i]:
                sum_TA = sum_TA + df["TA_L"].iloc[i]
            sum = sum + df["PL_L"].iloc[i] + df["TA_L"].iloc[i]
        CI_L.loc[attempt_num-1, taskname] = (2 * (sum_PL + sum_TA) / sum) * 100
    
    # 効き足側を0に格納
    if g.domi_leg[ID] == 0:
        result_np_summary[ID, :, 0] = np.array(CI_R.mean(axis=0, skipna=True))
        result_np_summary[ID, :, 1] = np.array(CI_L.mean(axis=0, skipna=True))
        CI_std[ID, :, 0] = np.array(CI_R.std(axis = 0, skipna=True))
        CI_std[ID, :, 1] = np.array(CI_L.std(axis = 0, skipna=True))        
    elif g.domi_leg[ID] == 1:
        result_np_summary[ID, :, 0] = np.array(CI_L.mean(axis=0, skipna=True))
        result_np_summary[ID, :, 1] = np.array(CI_R.mean(axis=0, skipna=True))
        CI_std[ID, :, 0] = np.array(CI_L.std(axis = 0, skipna=True))
        CI_std[ID, :, 1] = np.array(CI_R.std(axis = 0, skipna=True))
    
    CI_R = pd.concat([CI_R, CI_L])
    
    excel_output(CI_R, output_name, sheet_name)
    
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
    ax.bar(x, result_np_summary[ID, :, 0], width=0.5, yerr=err, capsize=3, label = g.task, color=color_map)
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
    ax.bar(x, result_np_summary[ID, :, 1], width=0.5, yerr=err, capsize=3, label = g.task, color=color_map)
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
result_domi = pd.DataFrame(result_np_summary[[0,2,3,4,5], :, 0], columns=g.task)
std_domi = pd.DataFrame(CI_std[[0,2,3,4,5], :, 0], columns=g.task)

result_nondomi = pd.DataFrame(result_np_summary[[0,2,3,4,5], :, 1], columns=g.task)
std_nondomi = pd.DataFrame(CI_std[[0,2,3,4,5], :, 1], columns=g.task)

# 新しいファイルに結果を書き込む。
new_file_path = os.path.join(output_dir, "CI_ancle_domi.csv")
result_domi.to_csv(new_file_path)
new_file_path = os.path.join(output_dir, "CI_ancle_nondomi.csv")
result_nondomi.to_csv(new_file_path)

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
    err = [std_domi.iloc[:, i]]
    ax.bar(sub_num+slide, result_domi.iloc[:,i], width=0.12
           , yerr=err, capsize=3, label = g.task[i])
ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
ax.set_xticks(sub_num + 0.3)
ax.set_ylim([0, 120])
ax.set_ylabel("Co-contraction index of\ndominant foot(%)")
ax.set_xticklabels(sublist)
#plt.show()
output_filename = "/plot_comparison_CI_ancle_domi.png"
fig.savefig(output_dir + output_filename)
plt.close()

###
### グラフをプロット　横並び
### 非利き足
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(task_num):
    slide = i*0.15
    err = [std_nondomi.iloc[:, i]]
    ax.bar(sub_num+slide, result_nondomi.iloc[:,i], width=0.12
           , yerr=err, capsize=3, label = g.task[i])
ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
ax.set_xticks(sub_num + 0.3)
ax.set_ylim([0, 120])
ax.set_ylabel("Co-contraction index of\nnondominant foot(%)")
ax.set_xticklabels(sublist)
#plt.show()
output_filename = "/plot_comparison_CI_ancle_nondomi.png"
fig.savefig(output_dir + output_filename)
plt.close()

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


##############################
#
# 全体の結果を作成
#
##############################
mean_result_domi = result_domi.mean(axis = 0)
mean_std_domi = result_domi.std(axis = 0)
mean_result_nondomi = result_nondomi.mean(axis = 0)
mean_std_nondomi = result_nondomi.std(axis = 0)

# グラフをプロット
# 効き足
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12   

x = np.arange(1, task_num+1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
err = [mean_std_domi]
ax.bar(x, mean_result_domi, width=0.5, yerr=err, capsize=3, label = g.task, color=color_map)
#    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
#    ax.set_xticks(index_num + 0.3)
ax.set_ylim([0, 120])
ax.set_ylabel("Co-contraction index of\ndominant foot(%)")
ax.set_xticks(x)
ax.set_xticklabels(g.task)

#plt.show()
output_filename = "/plot_mean_CI_ancle_domi.png"
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
err = [mean_std_nondomi]
ax.bar(x, mean_result_nondomi, width=0.5, yerr=err, capsize=3, label = g.task, color=color_map)
#    ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
#    ax.set_xticks(index_num + 0.3)
ax.set_ylim([0, 120])
ax.set_ylabel("Co-contraction index of\nnondominant foot(%)")
ax.set_xticks(x)
ax.set_xticklabels(g.task)

#plt.show()
output_filename = "/plot_mean_CI_ancle_nondomi.png"
fig.savefig(output_dir + output_filename)
plt.close()

