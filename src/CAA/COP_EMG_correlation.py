# %%
# -*- coding: utf-8 -*-
# EMGとCOPの相関係数を算出するコード

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate, savgol_filter

import global_value as g

task_list = ["NC", "FB", "DB"]
sampling = 2000
task_num = len(g.task)
# 除外する被験者番号
exclude_subjects = [1, 3, 8, 9, 10]# DBgroup
#exclude_subjects = [3, 5, 7]# FBgroup
#exclude_subjects = []# whole

def main():
    output_dir, output_name, plot_dir = output_preparation()
    all_results = []

    for ID in range(g.subnum):
        sheet_name = f"Correlation_sub{ID+1}"
        df_id = calculate_correlation(ID, plot_dir)  # 戻り値としてDataFrameを受け取り
        if df_id is not None:
            # IDごとの結果シートへ出力
            excel_output(df_id, output_name, sheet_name)
            all_results.append(df_id)

    # 全被験者をまとめて、被験者番号3, 5, 7を除外
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all = df_all[~df_all["subject"].isin(exclude_subjects)]
        df_all = df_all.sort_values("task").reset_index(drop=True)
        excel_output(df_all, output_name, "AllData")

def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_CAA/COP_EMG/DBgroup" % (g.datafile)
    output_name = output_dir + "/result.xlsx"
    plot_dir = output_dir + "/plot"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    # エクセルファイルの初期化
    if os.path.isfile(output_name):
        os.remove(output_name)

    return output_dir, output_name, plot_dir

def calculate_correlation(ID, plot_dir):
    cop_cols = ["ax","ay"]
    emg_cols = ["TA_R","TA_L","PL_R","PL_L","SO_R","SO_L","MF_R","MF_L","IO_R","IO_L"]
    emg_file_list = preparation_emg(ID)
    results = []
    max_lag_samples = 250  # 500ms(=0.5s)×500Hz=250サンプル

    def downsample_with_rms(data, original_fs, target_fs):
        # ローパスフィルタ
        data = lowpass_filter(data, 50, original_fs)
        down_factor = original_fs // target_fs
        window = int((original_fs / 10) / 2)
        slide = int(window / 25)  # スライド幅をウィンドウ幅の1/10に設定
        result = []
        for i in range(0, len(data), slide):
            if i < window:
                result.append(np.sqrt(np.square(data[:i+window]).mean()))
            elif (len(data) - i) < window:
                result.append(np.sqrt(np.square(data[i-window:]).mean()))
            else:
                result.append(np.sqrt(np.square(data[i-window:i+window]).mean()))
        return np.array(result)
    """
    def lowpass_and_savfilter(data, original_fs):
        # ローパスフィルタ
        data = lowpass_filter(data, 50, original_fs)
        # サビツキー・ゴレイフィルタ
        window_length = min(51, len(data) // 2 * 2 + 1)  # データ長に合わせて調整
        polyorder = 3
        filtered = savgol_filter(data, window_length, polyorder)
        return filtered
    """

    def standardize(data):
        return (data - np.mean(data)) / np.std(data)

    def plot_correlation(corr, lags, task, subject, attempt, plot_dir):
        plt.figure()
        plt.plot(lags, corr)
        plt.xlabel("Lag (ms)")
        plt.ylabel("Correlation coefficient")
        plt.title(f"Task: {task}, Subject: {subject}, Attempt: {attempt}")
        plt.grid(True)
        filename = f"/correlation_task_{task}_subject_{subject}_attempt_{attempt}.png"
        plt.savefig(plot_dir + filename)
        plt.close()

    for emg_file in emg_file_list:
        # タスク名は先頭2文字、試行番号はその後の4桁
        task = emg_file[-10:-8]     # 例: "D1"
        attempt_str = emg_file[-6:-4]  # 例: "00" + "01" -> "0001"
        try:
            attempt = int(attempt_str)
        except:
            continue

        # COPファイルパターン (subject: ID+1, task, attempt)
        sub_str = f"sub{ID+1:02d}"
        cop_filename = f"{sub_str}{task}{attempt:02d}.csv"
        cop_file = os.path.join(f"D:/User/kanai/Data/{g.datafile}/result_COP/dump", cop_filename)
        if not os.path.isfile(cop_file):
            continue

        emg_df = pd.read_csv(emg_file)
        cop_df = pd.read_csv(cop_file)
        # COPは1000Hzから500Hzにダウンサンプリング
        downsampled_cop = {}
        for c in cop_cols:
            if c not in cop_df.columns:
                break
            #cop_df[c] = lowpass_and_savfilter(cop_df[c], 1000)
            downsampled_cop[c] = downsample_with_rms(cop_df[c], 1000, 500)
            downsampled_cop[c] = standardize(downsampled_cop[c])
        cop_df = pd.DataFrame(downsampled_cop)
        
        for m in emg_cols:
            if m not in emg_df.columns:
                continue
            emg_df[m] = standardize(emg_df[m])

        row_dict = {"task": task, "subject": ID+1}
        # 空の相関/ラグ用辞書を準備
        r_dict = {}
        lag_dict = {}
        for c in cop_cols:
            if c not in cop_df.columns:
                continue
            for m in emg_cols:
                if m not in emg_df.columns:
                    continue
                # クロスコリレーションで最大相関とラグを求める
                corr_full = correlate(emg_df[m], cop_df[c], mode="full")
                lags = np.arange(- (len(cop_df[c]) - 1), len(emg_df[m]))
                # 0～+500ms のみ使用
                valid_idx = np.where((lags >= 0) & (lags <= max_lag_samples))[0]
                if not len(valid_idx):
                    continue
                corr_sub = corr_full[valid_idx] / (np.std(emg_df[m]) * np.std(cop_df[c]) * len(emg_df[m]))
                idx_local = np.argmax(np.abs(corr_sub))
                idx_max = valid_idx[idx_local]
                max_corr = corr_full[idx_max] / (np.std(emg_df[m]) * np.std(cop_df[c]) * len(emg_df[m]))
                best_lag = lags[idx_max]
                # 列名を生成
                col_r = f"r_{c}-{m}"
                col_lag = f"Lag_{c}-{m}"
                r_dict[col_r] = max_corr
                lag_dict[col_lag] = best_lag

                # 相関係数のグラフをプロット
                #plot_correlation(corr_sub, lags[valid_idx], task, ID+1, attempt, plot_dir)
        
        # 相関列とラグ列の間に1列空けるため、まず相関列群→ダミー列→ラグ列群に分割
        row_dict.update(r_dict)
        row_dict[""] = ""  # ダミー列
        row_dict.update(lag_dict)
        results.append(row_dict)

    if not results:
        return None
    df_results = pd.DataFrame(results)
    return df_results

def preparation_emg(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC/*.csv" % (g.datafile, ID + 1)
    file_list = glob.glob(input_dir)

    return file_list

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def excel_output(data, output_name, sheet_name):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name)

def plt_whole(mean, std, output_dir, output_filename, ylabel):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    x = np.arange(1, len(task_list) + 1)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    err = [std]
    ax.bar(x, mean, width=0.5, yerr=err, capsize=3, label=task_list)
    ax.tick_params(direction="in")
    ax.set_ylim([-1, 1])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)

    fig.savefig(output_dir + output_filename)
    plt.close()

if __name__ == "__main__":
    main()

# %%
