# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import welch, butter, filtfilt, windows
import matplotlib.pyplot as plt
import global_value as g

def main():
    # sampling_rate[Hz]
    f_s = 1000
    output_name, output_dir_plot = output_preparation()
    for ID in range(g.subnum):
        sheet_name = f"sub{ID+1}"
        # ルートフォルダのパスを指定
        file_list = preparation()
        sub_name = "sub%.2d" %(ID+1)

        # 周波数/PSD を一時保管する辞書を用意
        freq_masked = None
        psd_x_dict = {}
        psd_y_dict = {}

        # リストの順に呼び出し
        sub_list = [s for s in file_list if sub_name in s]
        for t in g.task:
            task_list = [s for s in sub_list if t in s]
            for f in task_list:
                # CSVファイルを開く
                df = pd.read_csv(f)
                attempt_num = int(f[(f.find("\\")+9)])
                # 変更: DCオフセットを引いてからPSDを計算
                #sig_x = df.iloc[:,0] - df.iloc[:,0].mean()
                #sig_y = df.iloc[:,1] - df.iloc[:,1].mean()
                # ハイパスによるDC成分の除去
                cutoff_frequency = 0.1  # 0.1Hz以上の成分を通過させる
                sig_x = highpass_filter(df.iloc[:,0], cutoff_frequency, f_s)
                sig_y = highpass_filter(df.iloc[:,1], cutoff_frequency, f_s)
                # ハニング窓を適用
                window = windows.hann(len(sig_x))
                sig_x_windowed = sig_x * window
                sig_y_windowed = sig_y * window
                
                # ウェルチで周波数解析
                freqs_x, psd_x = welch(sig_x_windowed, fs=f_s, nperseg=20000)
                freqs_y, psd_y = welch(sig_y_windowed, fs=f_s, nperseg=20000)

                # 周波数 10 Hz 以下に限定
                mask_x = freqs_x <= 10
                mask_y = freqs_y <= 10
                # 同じ配列なら freq_x のみ保存
                if freq_masked is None:
                    freq_masked = freqs_x[mask_x]

                # 辞書にタスク＋試行番号をキーとして保存
                key_x = f"{t}{attempt_num}"  # 例: NC1, FB2 など
                key_y = f"{t}{attempt_num}"
                psd_x_dict[key_x] = psd_x[mask_x]
                psd_y_dict[key_y] = psd_y[mask_y]
                
                plot_fft(freqs_x, psd_x, freqs_y, psd_y, output_dir_plot, ID, attempt_num, t)

        # 上記をまとめてデータフレーム化
        df_result = build_psd_dataframe(freq_masked, psd_x_dict, psd_y_dict)
        excel_output(df_result, output_name, sheet_name)


def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y
"""
周波数が 10Hz 以下のデータを抜き出し、X/Y を区切り列付きで同じシートに書き込む
"""
def build_psd_dataframe(freq_array, psd_x_dict, psd_y_dict):
    # 1) psd_x_dict.keys() から試行番号だけ取り出して昇順に並べる
    attempts = set()
    for k in psd_x_dict.keys():
        at_str = "".join([c for c in k if c.isdigit()])
        attempts.add(int(at_str))
    attempts = sorted(attempts)

    # 2) g.task の順でキー (タスク + 試行番号) を組み立てる
    my_keys = []
    for t in g.task:
        for a in attempts:
            key = f"{t}{a}"
            if key in psd_x_dict:
                my_keys.append(key)

    # 3) freq 列 + X データ列 + 空白列 + Y データ列
    columns = ["freq"] + my_keys + [""] + my_keys
    nrows = len(freq_array)
    ncols = len(columns)
    data = np.full((nrows, ncols), np.nan)

    # 最左列に freq を格納
    data[:, 0] = freq_array

    # psd_x_dict を該当列へコピー
    for idx, k in enumerate(my_keys, start=1):
        data[:, idx] = psd_x_dict[k]

    # 空白列を挟んで psd_y
    blank_col = 1 + len(my_keys)
    for idx, k in enumerate(my_keys, start=blank_col + 1):
        data[:, idx] = psd_y_dict[k]

    df = pd.DataFrame(data, columns=columns)
    return df


""" 
周波数軸に対してPSDをプロットして保存
"""
def plot_fft(freqs_x, psd_x, freqs_y, psd_y, output_dir_plot, ID, attempt_num, task):
    plt.figure()
    plt.plot(freqs_x, psd_x, label="X-axis PSD")
    plt.plot(freqs_y, psd_y, label="Y-axis PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.xlim(0, 3.0)
    plt.ylim(0, 500000)
    plt.legend()
    filename = f"/plot_sub{ID+1}_{task}_attempt{attempt_num}.png"
    plt.savefig(output_dir_plot + filename)
    plt.close()


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
    output_dir = "D:/User/kanai/Data/%s/result_FFT/COP" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_FFT/COP/plot" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_name, output_dir_plot


"""
周波数が 10Hz 以下のデータを抜き出し、X/Y を区切り列付きで同じシートに書き込む
"""
def store_psd_in_excel(freqs_x, psd_x, freqs_y, psd_y, attempt_num, output_name, sheet_name):
    mask_x = freqs_x <= 10
    mask_y = freqs_y <= 10
    df_x = pd.DataFrame({
        "freq_x": freqs_x[mask_x],
        "psd_x": psd_x[mask_x],
    })
    # 空白用列を挟んで Y 軸
    df_y = pd.DataFrame({
        "freq_y": freqs_y[mask_y],
        "psd_y": psd_y[mask_y],
    })
    # 列方向で結合 (空列を用意)
    blank_col = pd.DataFrame(np.full((len(df_x), 1), np.nan), columns=[""])
    df_combined = pd.concat([df_x, blank_col, df_y], axis=1)

    startrow = (attempt_num - 1) * (len(df_combined) + 2)
    # 変更: excel_output を利用して startrow を指定し、ヘッダは付けない
    excel_output(df_combined, output_name, sheet_name, startrow=startrow, header=False)

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
