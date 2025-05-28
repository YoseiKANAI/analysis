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

task_list = ["NC", "FB", "D1", "D2", "DW"]  # 5タスクに拡張
muscle_groups = ["TA", "PL", "SO", "GM", "MF", "IO"]
color_map = [cm.tab10(i) for i in range(len(task_list))]

def main():
    output_dir, output_name = output_preparation()
    exclude_subjects = []
    all_rows = []

    # すべてのCV値を一時保存: {subj: {muscle: {"NC": [..], "FB": [..], ...}}}
    all_cv = {}
    for ID in range(g.subnum):
        if ID in exclude_subjects:
            continue
        sub_str = f"sub{ID+1:02d}"
        subj_cv = {muscle: {t: [] for t in task_list} for muscle in muscle_groups}
        for t in task_list:
            for file_idx, (cv_domi, cv_non_domi) in enumerate(calc_cv_all(ID, t)):
                for i, muscle in enumerate(muscle_groups):
                    # domi, non_domi両方保存
                    subj_cv[muscle][t].append((cv_domi[i], cv_non_domi[i]))
        all_cv[sub_str] = subj_cv

    # 正規化して出力
    for ID in range(g.subnum):
        if ID in exclude_subjects:
            continue
        sub_str = f"sub{ID+1:02d}"
        subject_rows = []
        subj_cv = all_cv[sub_str]
        # 各筋肉ごとにNCのdomi/non_domi平均を計算
        nc_mean = {}
        for muscle in muscle_groups:
            nc_domi_vals = [v[0] for v in subj_cv[muscle]["NC"] if v[0] is not None and not np.isnan(v[0])]
            nc_non_domi_vals = [v[1] for v in subj_cv[muscle]["NC"] if v[1] is not None and not np.isnan(v[1])]
            nc_mean[muscle] = {
                "domi": np.mean(nc_domi_vals) if len(nc_domi_vals) > 0 else np.nan,
                "non_domi": np.mean(nc_non_domi_vals) if len(nc_non_domi_vals) > 0 else np.nan,
            }
        # 各タスク・ファイルごとに正規化値を計算
        for t in task_list:
            n_files = len(subj_cv[muscle_groups[0]][t])
            for file_idx in range(n_files):
                row = {"Subject": sub_str, "Task": t, "FileIndex": file_idx+1}
                for muscle in muscle_groups:
                    domi, non_domi = subj_cv[muscle][t][file_idx] if file_idx < len(subj_cv[muscle][t]) else (np.nan, np.nan)
                    norm_domi = domi / nc_mean[muscle]["domi"] if nc_mean[muscle]["domi"] not in [0, np.nan] else np.nan
                    norm_non_domi = non_domi / nc_mean[muscle]["non_domi"] if nc_mean[muscle]["non_domi"] not in [0, np.nan] else np.nan
                    row[f"{muscle}_domi"] = norm_domi
                    row[f"{muscle}_non_domi"] = norm_non_domi
                subject_rows.append(row)
        df_sub = pd.DataFrame(subject_rows, columns=["Subject", "Task", "FileIndex"] + [f"{m}_domi" for m in muscle_groups] + [f"{m}_non_domi" for m in muscle_groups])
        excel_output(df_sub, output_name, sub_str)
        all_rows.extend(subject_rows)

    # ALLシート
    df_all = pd.DataFrame(all_rows, columns=["Subject", "Task", "FileIndex"] + [f"{m}_domi" for m in muscle_groups] + [f"{m}_non_domi" for m in muscle_groups])
    df_all["Task"] = pd.Categorical(df_all["Task"], categories=task_list, ordered=True)
    df_all_sorted = df_all.sort_values(["Task", "Subject", "FileIndex"]).reset_index(drop=True)
    excel_output(df_all_sorted, output_name, "ALL")

def calc_cv_all(ID, task):
    """
    指定被験者・タスクの全ファイルについて、全筋肉のCV値（domi, non_domi）リストを返す
    戻り値: [(cv_domi_list, cv_non_domi_list), ...]  # ファイルごと
    """
    input_dir = f"D:/User/kanai/Data/{g.datafile}/sub{ID+1}/csv/MVC/"
    files = glob.glob(os.path.join(input_dir, f"*{task}*.csv"))
    results = []
    for f in files:
        df = pd.read_csv(f)
        cv_domi_list = []
        cv_non_domi_list = []
        for muscle in muscle_groups:
            if (muscle + "_R" not in df.columns) or (muscle + "_L" not in df.columns):
                cv_domi_list.append(np.nan)
                cv_non_domi_list.append(np.nan)
                continue
            R = df[muscle + "_R"]
            L = df[muscle + "_L"]
            if g.domi_leg[ID] == 0:
                domi = R.std(skipna=True) / R.mean(skipna=True) if R.mean(skipna=True) != 0 else np.nan
                nondomi = L.std(skipna=True) / L.mean(skipna=True) if L.mean(skipna=True) != 0 else np.nan
            else:
                domi = L.std(skipna=True) / L.mean(skipna=True) if L.mean(skipna=True) != 0 else np.nan
                nondomi = R.std(skipna=True) / R.mean(skipna=True) if R.mean(skipna=True) != 0 else np.nan
            cv_domi_list.append(domi)
            cv_non_domi_list.append(nondomi)
        results.append((cv_domi_list, cv_non_domi_list))
    return results

def output_preparation():
    output_dir = f"D:/User/kanai/Data/{g.datafile}/result_EMG/CV/whole"
    output_name = output_dir + "/result.xlsx"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(output_name):
        os.remove(output_name)
    return output_dir, output_name

def excel_output(data, output_name, sheet_name):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a", if_sheet_exists="replace") as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    main()
