# %%
# -*- coding: utf-8 -*-
"""
COPデータに対してDFA (Detrended fluctuation analysis) を行い，
各被験者ごとにスケーリング指数 (α) をExcelにまとめるとともに，
各ファイルのDFA結果（log-logプロット）を出力するプログラム

参考: FFT_COP.py の入力／出力関数を流用
@author: ShimaLab
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import global_value as g

def main():
    sampling = 1000
    # ウィンドウサイズを対数のスケールで作成
    scales = np.unique(np.logspace(np.log10(100), np.log10(10000), num=20, base=10).astype(int))
    file_list = preparation()
    output_excel, output_dir_plot = output_preparation()

    df_all = []
    #exclude_subjects = [3, 5, 7]  # 除外する被験者番号
    exclude_subjects = [1, 3, 8, 9, 10]
    for ID in range(g.subnum):
        if ID + 1 in exclude_subjects:
            continue
        sub_str = "sub%.2d" % (ID+1)
        sub_list = [f for f in file_list if sub_str in f]
        row_list = []
        # タスクg.taskの順序で解析
        for t in g.task:
            task_list = [s for s in sub_list if t in s]
            for f in task_list:
                df = pd.read_csv(f)
                attempt_num = int(f[(f.find("\\")+9)])
                if ("ax" not in df.columns) or ("ay" not in df.columns):
                    continue
                # ハイパスフィルタを適用
                signal_x = highpass_filter(df["ax"].values, 0.1, sampling)
                signal_y = highpass_filter(df["ay"].values, 0.1, sampling)
                # DFAを実行
                slope_x, *rest_x = dfa(signal_x, scales)
                slope_y, *rest_y = dfa(signal_y, scales)
                row_list.append({
                    "Subject": sub_str, 
                    "Task": t,
                    "Attempt": attempt_num,
                    "Alpha_x": slope_x,
                    "Alpha_y": slope_y
                })
        df_sub = pd.DataFrame(row_list, columns=["Subject","Task","Attempt","Alpha_x","Alpha_y"])
        excel_output(df_sub, output_excel, f"sub{ID+1}")
        df_all.append(df_sub)

    # すべての被験者データを結合後、そのまま AllTasks シートに出力 (275行)
    df_all = pd.concat(df_all, ignore_index=True)
    # 追加: Task列をカテゴリ化し、(Task, Attempt, Subject)でソート
    df_all["Task"] = pd.Categorical(df_all["Task"], categories=g.task, ordered=True)
    df_all_sorted = df_all.sort_values(["Task", "Attempt", "Subject"]).reset_index(drop=True)
    excel_output(df_all_sorted, output_excel, "AllTasks")

    # グラフ出力
    plot_grouped_correlation(df_all_sorted, output_dir_plot)

"""
DFAを実行する関数
"""
def dfa(x, scales):
    """
    x: 1次元時系列
    scales: 調べるウィンドウサイズのリスト（例：logspaceで生成）
    returns: スケーリング指数α, scales, 各サイズでの平均残差 F(n) 等
    """
    # 平均を引いた累積和
    y = np.cumsum(x - np.mean(x))
    F = []
    for s in scales:
        n_windows = int(len(y) / s)
        if n_windows == 0:
            continue
        rms = []
        for j in range(n_windows):
            segment = y[j*s:(j+1)*s]
            x_axis = np.arange(s)
            # 線形フィッティング
            coeffs = np.polyfit(x_axis, segment, 1)
            trend = np.polyval(coeffs, x_axis)
            rms.append(np.sqrt(np.mean((segment - trend)**2)))
        F.append(np.sqrt(np.mean(np.array(rms)**2)))
    # 対応するウィンドウサイズ（上位分だけ）
    scales_arr = np.array(scales[:len(F)])
    F_arr = np.array(F)
    log_scales = np.log(scales_arr)
    log_F = np.log(F_arr)
    slope, intercept = np.polyfit(log_scales, log_F, 1)
    return slope, scales_arr, F_arr, log_scales, log_F

def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)
    
"""
COPファイルのパスリストを作成
"""
def preparation():
    # フォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/result_COP/dump/*.csv" %(g.datafile)
    file_list = glob.glob(root_dir)
    
    return file_list


"""
ファイル名の定義
"""        
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_COP/DFA/DBgroup" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_COP/DFA/FBgroup/plot" %(g.datafile)
    output_excel = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_excel)):
        os.remove(output_excel)
    
    return output_excel, output_dir_plot

""" 
周波数軸に対してPSDをプロットして保存
"""
def plot_raw(axis_x, x, output_dir_plot, ID, attempt_num, task):
    plt.figure()
    plt.plot(axis_x, x)
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("PSD")
    #plt.xlim(0, 3.0)
    plt.ylim(0, 10)
    plt.legend()
    filename = f"/plot_sub{ID+1}_{task}_attempt{attempt_num}.png"
    plt.savefig(output_dir_plot + filename)
    plt.close()
    

"""
excelに出力
"""
def excel_output(data, output_name, sheet_name, startrow=0, header=True):
    """startrow や header を指定して出力できるように変更"""
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a", if_sheet_exists="overlay") as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow,
                          header=header, index=False)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow,
                          header=header, index=False)

def plot_grouped_correlation(df_all_sorted, output_dir_plot):
    plt.figure(figsize=(8,4))
    for task in df_all_sorted["Task"].unique():
        df_task = df_all_sorted[df_all_sorted["Task"] == task]
        plt.errorbar(df_task["Attempt"], df_task["Alpha_x"], yerr=df_task["Alpha_x"].std(), fmt='o', label=f"{task} Alpha_x")
        plt.errorbar(df_task["Attempt"], df_task["Alpha_y"], yerr=df_task["Alpha_y"].std(), fmt='x', label=f"{task} Alpha_y")
    plt.xlabel("Attempt")
    plt.ylabel("Alpha")
    plt.legend()
    save_path = os.path.join(output_dir_plot, "grouped_correlation.png")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
