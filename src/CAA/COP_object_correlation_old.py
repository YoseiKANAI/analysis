# %%
# COPとobjectの相互相関を計算し、Excelに出力するスクリプト
# old
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, resample, butter, filtfilt
import global_value as g

def main():
    output_dir, output_excel = output_preparation()
    all_results = []
    # 被験者ごとに処理（被験者番号は g.subnum を使用）
    for ID in range(1, g.subnum+1):
        sub_id = f"sub{ID:02d}"
        # COPファイルは result_COP/dump 配下
        cop_pattern = f"D:/User/kanai/Data/{g.datafile}/result_COP/dump/*{sub_id}*.csv"
        cop_files = glob.glob(cop_pattern)
        # object は各被験者フォルダ内に格納
        obj_folder = f"D:/User/kanai/Data/{g.datafile}/sub{ID}/csv/object_absolute/"#絶対座標でのobjectデータ
        results_sub = []
        for cop_file in cop_files:
            # 例: COPファイル名 "sub01D101.csv"
            filename = os.path.basename(cop_file)
            try:
                # 被験者番号は先頭5文字, 次2文字がタスク, その後の数字が試行番号
                # 例："sub01D101.csv" -> subject="sub01", task="D1", attempt="01"
                task = filename[5:7]                # "D1"
                attempt = int(filename[7:-4])       # "01" -> 1
            except Exception as e:
                continue

            cop_df = pd.read_csv(cop_file)
            # COPデータは1000Hz; x方向とy方向を使用
            cop_signals = {
                "ax": cop_df["ax"].values,
                "ay": cop_df["ay"].values
            }

            # objectファイルは被験者フォルダ内，ファイル名は "D10001.csv" など
            # objectファイル名の先頭2文字がタスク、残りの数字が試行番号
            obj_file = os.path.join(obj_folder, f"{task}{attempt:04d}.csv")
            if not os.path.isfile(obj_file):
                continue  # 対応するobjectファイルが存在しない場合はスキップ
            obj_df = pd.read_csv(obj_file)
            # objectは100Hz; x方向とy方向を使用
            obj_signals = {
                "X": obj_df["X"].values,
                "Y": obj_df["Y"].values
            }

            row_dict = {"Subject": sub_id, "Task": task, "Attempt": attempt}
            for cop_key, cop_signal in cop_signals.items():
                for obj_key, obj_signal in obj_signals.items():
                    # 0.3Hz以下のローパスフィルタを適用
                    cop_signal_downsampled = lowpass_filter(cop_signal, 0.3, 1000)
                    obj_signal_filtered = lowpass_filter(obj_signal, 0.3, 100)
                    
                    # COPデータを100Hzにダウンサンプリング
                    cop_signal_downsampled = downsample_with_rms(cop_signal_downsampled, 1000, 100)

                    # 信号を標準化（平均を引いて標準偏差で割る）
                    cop_signal_downsampled = (cop_signal_downsampled - np.mean(cop_signal_downsampled)) / np.std(cop_signal_downsampled)
                    obj_signal_filtered = (obj_signal_filtered - np.mean(obj_signal_filtered)) / np.std(obj_signal_filtered)

                    # 相互相関を計算し、正規化
                    corr = correlate(cop_signal_downsampled, obj_signal_filtered, mode="same")
                    corr /= np.sqrt(np.sum(cop_signal_downsampled**2) * np.sum(obj_signal_filtered**2))

                    # 0～500ms の範囲に限定
                    max_lag = 50  # 500ms
                    center = len(corr) // 2
                    lag_range = int(max_lag * (100 / 100))  # 500msをサンプル数に変換
                    limited_corr = corr[center:center + lag_range + 1]  # 正のラグのみ使用

                    max_corr = limited_corr.max()
                    lag = np.argmax(limited_corr) * (1000 / 100)  # ラグをms単位に変換

                    row_dict[f"MaxCorr_{cop_key}_{obj_key}"] = max_corr
                    row_dict[f"Lag_{cop_key}_{obj_key} (ms)"] = lag

                    # 任意：各ファイルごとの相互相関プロットの出力
                    #plot_correlation(cop_signal_downsampled, obj_signal_filtered, limited_corr, sub_id, task, attempt, output_dir, cop_key, obj_key)
            results_sub.append(row_dict)
        df_sub = pd.DataFrame(results_sub)
        excel_output(df_sub, output_excel, sub_id)
        all_results.append(df_sub)
    # 全被験者を結合し、タスク, 試行, 被験者順にソートして "AllSubjects" シートへ出力
    df_all = pd.concat(all_results, ignore_index=True)
    df_all["Task"] = pd.Categorical(df_all["Task"], categories=g.task, ordered=True)
    df_all_sorted = df_all.sort_values(["Task", "Attempt", "Subject"]).reset_index(drop=True)
    excel_output(df_all_sorted, output_excel, "AllSubjects")

def downsample_with_rms(data, original_fs, target_fs):
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

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def plot_correlation(cop_signal, obj_signal, corr, sub_id, task, attempt, out_dir, cop_key, obj_key):
    plt.figure(figsize=(8,4))
    plt.plot(corr, label="Cross-correlation")
    plt.title(f"{sub_id} {task}{attempt} {cop_key}-{obj_key}")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()
    save_path = os.path.join(out_dir, "plot")
    os.makedirs(save_path, exist_ok=True)
    filename = f"{sub_id}_{task}{attempt}_{cop_key}_{obj_key}_corr.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def output_preparation():
    out_dir = f"D:/User/kanai/Data/{g.datafile}/result_CAA/COP_obj"
    os.makedirs(out_dir, exist_ok=True)
    output_excel = os.path.join(out_dir, "result.xlsx")
    if os.path.isfile(output_excel):
        os.remove(output_excel)
    return out_dir, output_excel

def excel_output(data, output_name, sheet_name, startrow=0, header=True):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a", if_sheet_exists="overlay") as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow, header=header, index=False)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name, startrow=startrow, header=header, index=False)

if __name__ == "__main__":
    main()
