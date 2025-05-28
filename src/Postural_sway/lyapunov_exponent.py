# %%
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:41:24 2023

@author: ShimaLab
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.signal import butter, filtfilt
import global_value as g

def main():
    sampling = 1000
    file_list = preparation()
    output_excel, _ = output_preparation()

    df_all = []
    exclude_subjects = []
    for ID in range(g.subnum):
        if ID + 1 in exclude_subjects:
            continue
        sub_str = "sub%.2d" % (ID+1)
        sub_list = [f for f in file_list if sub_str in f]
        row_list = []
        for t in g.task:
            task_list = [s for s in sub_list if t in s]
            for f in task_list:
                df = pd.read_csv(f)
                attempt_num = int(f[(f.find("\\")+9)])
                if ("ax" not in df.columns) or ("ay" not in df.columns):
                    continue
                # COPデータをローパスフィルタで処理する
                signal_x = lowpass_filter(df["ax"].values, 20, sampling)
                signal_y = lowpass_filter(df["ay"].values, 20, sampling)
                # リアプノフ指数を算出する
                lyap_x = max_lyapunov_exponent(signal_x, sampling)
                lyap_y = max_lyapunov_exponent(signal_y, sampling)
                row_list.append({
                    "Subject": sub_str,
                    "Task": t,
                    "Attempt": attempt_num,
                    "Lyapunov_x": lyap_x,
                    "Lyapunov_y": lyap_y
                })
        df_sub = pd.DataFrame(row_list, columns=["Subject", "Task", "Attempt", "Lyapunov_x", "Lyapunov_y"])
        excel_output(df_sub, output_excel, f"sub{ID+1}")
        df_all.append(df_sub)

    df_all = pd.concat(df_all, ignore_index=True)
    df_all["Task"] = pd.Categorical(df_all["Task"], categories=g.task, ordered=True)
    df_all_sorted = df_all.sort_values(["Task", "Attempt", "Subject"]).reset_index(drop=True)
    excel_output(df_all_sorted, output_excel, "AllTasks")

def max_lyapunov_exponent(signal, sampling):
    delay = 10
    embed_dim = 3
    max_time = len(signal) // 2
    embedded = np.array([signal[i:len(signal)-(embed_dim-1)*delay+i:delay] for i in range(embed_dim)]).T
    distances = [euclidean(embedded[i], embedded[i+1]) for i in range(len(embedded)-1)]
    log_distances = np.log(distances)
    return np.mean(log_distances[:max_time])

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def preparation():
    root_dir = "D:/User/kanai/Data/%s/result_COP/dump/*.csv" % (g.datafile)
    file_list = glob.glob(root_dir)
    return file_list

def output_preparation():
    output_dir = "D:/User/kanai/Data/%s/result_COP/Lyapunov" % (g.datafile)
    output_excel = output_dir + "/result.xlsx"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(output_excel):
        os.remove(output_excel)
    return output_excel, output_dir

def excel_output(data, output_name, sheet_name, startrow=0, header=True):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a", if_sheet_exists="overlay") as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow, header=header, index=False)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow, header=header, index=False)

if __name__ == "__main__":
    main()