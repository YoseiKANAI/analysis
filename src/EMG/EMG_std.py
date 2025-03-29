# %%
# -*- coding: utf-8 -*-
# EMGの標準偏差を算出するコード
# TA
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

task_list = ["NC","FB", "D1"]
sampling = 2000
task_num = len(task_list)
color_map = [cm.tab10(i) for i in range(len(task_list))]

muscle_groups = ["TA", "PL", "SO", "GM", "MF", "IO"]

def main():        
    # 結果出力先を std ディレクトリ直下に統合
    output_dir, output_name = output_preparation()

    # FBgroup用のデータを除外
    #exclude_subjects = [2, 4, 6]
    # DBgroup用のデータを除外
    #exclude_subjects = [0, 2, 7, 8, 9]
    exclude_subjects = []
    valid_idx = 0
    
    for muscle in muscle_groups:
        sheet_name_domi = f"{muscle}_domi"
        sheet_name_non_domi = f"{muscle}_non_domi"
        
        EMG_std_domi = np.full((g.subnum, g.attempt, task_num), np.nan)
        EMG_std_non_domi = np.full((g.subnum, g.attempt, task_num), np.nan)
        
        valid_idx = 0
        for ID in range(g.subnum):
            # 変更: exclude_subjects 該当 ID の処理をスキップ
            if ID in exclude_subjects:
                continue
            cal_EMG_std(EMG_std_domi, EMG_std_non_domi, ID, valid_idx, output_name,
                        sheet_name_domi, sheet_name_non_domi, muscle)
            valid_idx += 1
        # 被験者番号3, 5, 7を除外
        EMG_std_domi = np.delete(EMG_std_domi, exclude_subjects, axis=0)
        EMG_std_non_domi = np.delete(EMG_std_non_domi, exclude_subjects, axis=0)

        # 0次元目と1次元目を連結して次元削減
        EMG_std_domi = EMG_std_domi.reshape(-1, task_num)
        EMG_std_non_domi = EMG_std_non_domi.reshape(-1, task_num)

        # attemptの次元で平均値と標準偏差を求める
        EMG_std_domi_mean = np.nanmean(EMG_std_domi, axis=0)
        EMG_std_domi_std = np.nanstd(EMG_std_domi, axis=0)
        EMG_std_non_domi_mean = np.nanmean(EMG_std_non_domi, axis=0)
        EMG_std_non_domi_std = np.nanstd(EMG_std_non_domi, axis=0)

        # データをDataFrame型に戻す
        mean_domi = pd.Series(EMG_std_domi_mean[:3], index=task_list)
        std_domi = pd.Series(EMG_std_domi_std[:3], index=task_list)
        mean_non_domi = pd.Series(EMG_std_non_domi_mean[:3], index=task_list)
        std_non_domi = pd.Series(EMG_std_non_domi_std[:3], index=task_list)

        # グラフをプロット
        output_filename_domi = f"/plot_std_{muscle}_domi.svg"
        ylabel_domi = f"Normalized EMG Standard Deviation ({muscle} - Dominant)"
        plt_whole(mean_domi, std_domi, output_dir, output_filename_domi, ylabel_domi)

        output_filename_non_domi = f"/plot_std_{muscle}_non_domi.svg"
        ylabel_non_domi = f"Normalized EMG Standard Deviation ({muscle} - Non-Dominant)"
        plt_whole(mean_non_domi, std_non_domi, output_dir, output_filename_non_domi, ylabel_non_domi)

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
    # std ディレクトリに一つだけ result.xlsx を作成
    output_dir = f"D:/User/kanai/Data/%s/result_EMG/std/whole" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    os.makedirs(output_dir, exist_ok=True)
    # ファイルは毎回作り直す(必要に応じて削除)
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
EMGの標準偏差を計算するメインの関数
"""
def cal_EMG_std(EMG_std_domi, EMG_std_non_domi, ID, valid_idx, output_name,
                sheet_name_domi, sheet_name_non_domi, muscle):
    result_std_domi = pd.DataFrame(np.full((g.attempt, len(task_list)), np.nan), columns=task_list)
    result_std_non_domi = pd.DataFrame(np.full((g.attempt, len(task_list)), np.nan), columns=task_list)
    
    file_list = preparation(ID)
    for f in file_list:
        df = pd.read_csv(f)
        taskname = f[(f.find("\\")+1):(f.find("\\")+3)]
        attempt_num = int(f[(f.find("\\")+6)])
        
        R = df[muscle + "_R"]
        L = df[muscle + "_L"]
        
        # 変更: muscle文字列に対して適切なサフィックスを付けて標準偏差を算出
        if g.domi_leg[ID] == 0:
            dominant_value   = R.std(skipna=True)
            nondominant_value = L.std(skipna=True)
        else:
            dominant_value   = L.std(skipna=True)
            nondominant_value = R.std(skipna=True)
            
        result_std_domi.loc[attempt_num-1, taskname] = dominant_value
        result_std_non_domi.loc[attempt_num-1, taskname] = nondominant_value
    
    NC_mean_domi = result_std_domi.iloc[:, 0].mean(axis=0, skipna=True)
    NC_mean_non_domi = result_std_non_domi.iloc[:, 0].mean(axis=0, skipna=True)
    
    EMG_std_domi_norm = result_std_domi.div(NC_mean_domi, axis=1)
    EMG_std_non_domi_norm = result_std_non_domi.div(NC_mean_non_domi, axis=1)

    EMG_std_domi[ID, :, :] = np.array(EMG_std_domi_norm.iloc[:, :3])
    EMG_std_non_domi[ID, :, :] = np.array(EMG_std_non_domi_norm.iloc[:, :3])

    # 変更: 被験者ごとのシート分割をやめ、シート名を"TA_domi"のようにし、
    # 参加者データはシート下に連結（startrow=ID*g.attempt）して書き込む
    excel_append(EMG_std_domi_norm, output_name, sheet_name_domi, valid_idx)
    excel_append(EMG_std_non_domi_norm, output_name, sheet_name_non_domi, valid_idx)

def excel_append(data, output_name, sheet_name, valid_idx):
    startrow = valid_idx * g.attempt  # 参加者IDに応じて下へ連結
    # 変更: result.xlsx が無い場合は新規作成し、ある場合は append
    if not os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="w") as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow,
                          header=False, index=False)
    else:
        with pd.ExcelWriter(output_name, mode="a", if_sheet_exists="overlay") as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow,
                          header=False, index=False)

"""
全体グラフをプロットする関数
"""
def plt_whole(mean, std, output_dir, output_filename, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18   

    x = np.arange(1, len(task_list)+1)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1,1,1)
    ax.bar(x, mean, width=0.5, yerr=std, capsize=3, label = task_list, color=color_map)
    ax.tick_params(direction="in")
    ax.set_ylim([0, 2.1])
    #ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)

    fig.savefig(output_dir + output_filename)
    plt.close()

if __name__ == "__main__":
    main()
