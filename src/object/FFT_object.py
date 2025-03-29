# %%
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import pandas as pd
import numpy as np
#from scipy import signal
from scipy.signal import welch, butter, filtfilt, windows
import matplotlib.pyplot as plt
import global_value as g
import glob


def main():
    f_s = 100   # sampling frequency 100Hz
    output_name, output_dir_plot = output_preparation()
    for ID in range(g.subnum):
        file_list = preparation(ID)
        freq_masked = None
        psd_dict = {"X":{}, "Y":{}, "Z":{}}
        for csv_file in file_list:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file).replace(".csv","")
            # task名を抽出（例: ファイル名内の "\\" の直後2文字）
            task_name = csv_file[(csv_file.find("\\")+1):(csv_file.find("\\")+3)]
            if task_name == "NC":
                continue  # NCは出力しない
            attempt_num = int(csv_file[(csv_file.find("\\")+6)])  # 試行番号

            # ３軸について一度にFFT計算
            cutoff_frequency = 0.1
            sig_x = highpass_filter(df["X"], cutoff_frequency, f_s)
            sig_y = highpass_filter(df["Y"], cutoff_frequency, f_s)
            sig_z = highpass_filter(df["Z"], cutoff_frequency, f_s)
            window = windows.hann(len(sig_x))
            sig_x_win = sig_x * window
            sig_y_win = sig_y * window
            sig_z_win = sig_z * window
            freqs, psd_x = welch(sig_x_win, fs=f_s, nperseg=819)
            _, psd_y = welch(sig_y_win, fs=f_s, nperseg=819)
            _, psd_z = welch(sig_z_win, fs=f_s, nperseg=819)
            mask = freqs <= 10
            if freq_masked is None:
                freq_masked = freqs[mask]
            key = f"{task_name}{attempt_num}"  # 例: "FB1", "D11", etc.
            psd_dict["X"][key] = psd_x[mask]
            psd_dict["Y"][key] = psd_y[mask]
            psd_dict["Z"][key] = psd_z[mask]
            
            # ↓ 新規関数を呼び出し、３軸を１つのグラフにプロット
            plot_all_fft(freqs, psd_x, psd_y, psd_z, output_dir_plot, ID, attempt_num, task_name, filename)
        # DataFrame作成（後続はそのまま）
        df_result = build_psd_dataframe(freq_masked, psd_dict)
        excel_output(df_result, output_name, f"sub{ID+1}")

def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

"""
globを使ってフォルダ内のCSVファイル一覧を取得する関数
"""
def preparation(ID):
    root_dir = f"D:/User/kanai/Data/{g.datafile}/sub{ID+1}/csv/object/*.csv"
    file_list = glob.glob(root_dir, recursive=True)
    return file_list

"""
ファイル名の定義
"""        
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_FFT/object" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_FFT/object/plot" %(g.datafile)
    output_name = output_dir + "/result_FFT_object.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_name, output_dir_plot


"""
FFT結果をプロットする関数
"""
def plot_all_fft(freqs, psd_x, psd_y, psd_z, output_dir_plot, ID, attempt_num, task, filename):
    """３軸（X, Y, Z）のPSDを1枚のグラフにプロットする"""
    plt.figure()
    plt.plot(freqs, psd_x, label="X-axis")
    plt.plot(freqs, psd_y, label="Y-axis")
    plt.plot(freqs, psd_z, label="Z-axis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.xlim(0, 3.0)
    plt.ylim(0, 2000)
    plt.legend()
    plt.title(f"{filename}_{task}_attempt{attempt_num}")
    out_name = f"{output_dir_plot}/sub{ID+1}_{filename}_{task}_attempt{attempt_num}.png"
    plt.savefig(out_name)
    plt.close()

"""
PSDが格納されたDataFrameを作成する関数
"""
def build_psd_dataframe(freq_array, psd_dict):
    """
    まず、g.taskからNCを除いたタスクリスト（例: FB, D1, D2, DW）を取得し,
    各軸ごとにキー（例："FB1", "FB2", …）を試行番号昇順にソートして
    * Table1 (XとY)を作成：列は ["freq"] + keys_X + [""] + [keys_Y+"_Y"]
    * Table2 (Z)を作成：列は ["freq"] + [keys_Z+"_Z"]
    その後、Table1とTable2の間に1行の空行を挟んで垂直連結する。
    """
    # g.taskからNC以外のタスクの順序(入力順)を維持
    desired_tasks = [t for t in g.task if t != "NC"]
    axes = ["X", "Y", "Z"]
    keys_by_axis = {}
    for ax in axes:
        # psd_dict[ax]のキーは "task"+"attempt" の形式
        # キーを desired_tasks の順番に並べ替え，かつ試行番号昇順にソート
        all_keys = list(psd_dict[ax].keys())
        sorted_keys = []
        for t in desired_tasks:
            # tで始まるキーを抽出し、試行番号でソート
            keys_t = [k for k in all_keys if k.startswith(t)]
            keys_t = sorted(keys_t, key=lambda x: int(''.join(filter(str.isdigit, x))))
            sorted_keys.extend(keys_t)
        keys_by_axis[ax] = sorted_keys

    # Table1: Columns: ["freq"] + keys_X + [""] + [keys_Y+"_Y"]
    cols_table1 = ["freq"] + keys_by_axis["X"] + [""] + [k + "_Y" for k in keys_by_axis["Y"]]
    nrows = len(freq_array)
    data_table1 = np.full((nrows, len(cols_table1)), np.nan)
    data_table1[:,0] = freq_array
    # Fill X data
    for idx, k in enumerate(keys_by_axis["X"], start=1):
        data_table1[:, idx] = psd_dict["X"][k]
    # Fill Y data starting at column = 1+len(keys_by_axis["X"])+1
    startY = 1 + len(keys_by_axis["X"]) + 1
    for idx, k in enumerate(keys_by_axis["Y"], start=startY):
        data_table1[:, idx] = psd_dict["Y"][k]
    df_table1 = pd.DataFrame(data_table1, columns=cols_table1)
    
    # Table2: Columns: ["freq"] + [keys_Z+"_Z"] ; for Z axis.　
    cols_table2 = ["freq"] + [k + "_Z" for k in keys_by_axis["Z"]]
    data_table2 = np.full((nrows, len(cols_table2)), np.nan)
    data_table2[:,0] = freq_array
    for idx, k in enumerate(keys_by_axis["Z"], start=1):
        data_table2[:, idx] = psd_dict["Z"][k]
    df_table2 = pd.DataFrame(data_table2, columns=cols_table2)
    
    # Combine Table1 and Table2 vertically, with one blank row between
    blank_row = pd.DataFrame([[""]*len(df_table2.columns)], columns=df_table2.columns)
    df_result = pd.concat([df_table1, blank_row, df_table2], axis=0, ignore_index=True)
    return df_result


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

if __name__ == "__main__":
    main()