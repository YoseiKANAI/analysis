# %%
import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import global_value as g

def main():
    output_dir, output_name = output_preparation()
    all_results = []

    for ID in range(g.subnum):
        sheet_name = f"VelocityMeanStd_sub{ID+1}"
        df_id = calculate_velocity_mean_std(ID)  # 戻り値としてDataFrameを受け取り
        if df_id is not None:
            # IDごとの結果シートへ出力
            excel_output(df_id, output_name, sheet_name)
            all_results.append(df_id)

    # 全被験者をまとめて出力
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all = df_all.sort_values(["task", "attempt", "subject"]).reset_index(drop=True)
        excel_output(df_all, output_name, "AllData")
        plot_task_means(df_all, output_dir)

def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_COP/velocity_mean_std" % (g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    # エクセルファイルの初期化
    if os.path.isfile(output_name):
        os.remove(output_name)
    return output_dir, output_name

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def calculate_velocity_mean_std(ID):
    cop_cols = ["ax", "ay"]
    results = []

    def calculate_velocity(data, fs):
        cop_filterd = lowpass_filter(data, 50, 1000)
        return np.diff(cop_filterd) * fs

    cop_file_list = preparation_cop(ID)
    nc_means = {col: [] for col in cop_cols}  # NCの平均値を格納する辞書
    for cop_file in cop_file_list:
        # ファイル名からタスク名と試行番号を抽出
        filename = os.path.basename(cop_file)
        task = filename[5:7]  # タスク名
        attempt_str = filename[7:9]  # 試行番号
        try:
            attempt = int(attempt_str)
        except:
            continue

        cop_df = pd.read_csv(cop_file)
        
        row_dict = {"task": task, "subject": ID+1, "attempt": attempt}
        for c in cop_cols:
            if c not in cop_df.columns:
                continue
            # COP変位データを速度データに変換
            cop_velocity = calculate_velocity(cop_df[c].values, 1000)
            # 速度の平均値と標準偏差を算出
            mean_velocity = np.mean(np.abs(cop_velocity))
            std_velocity = np.std(np.abs(cop_velocity))
            # NCの平均値を格納
            if task == "NC":
                nc_means[c].append(mean_velocity)
            # 列名を生成
            col_mean_velocity = f"MeanVelocity_{c}"
            col_std_velocity = f"StdVelocity_{c}"
            row_dict[col_mean_velocity] = mean_velocity
            row_dict[col_std_velocity] = std_velocity
        
        results.append(row_dict)

    if not results:
        return None

    # NCの平均値を計算
    nc_mean_values = {c: np.mean(nc_means[c]) for c in cop_cols}

    # 正規化
    for result in results:
        for c in cop_cols:
            col_mean_velocity = f"MeanVelocity_{c}"
            if col_mean_velocity in result:
                result[col_mean_velocity] /= nc_mean_values[c]

    df_results = pd.DataFrame(results)
    return df_results

def preparation_cop(ID):
    # フォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/result_COP/dump/sub%.2d*.csv" % (g.datafile, ID + 1)
    file_list = glob.glob(input_dir)

    return file_list

def excel_output(data, output_name, sheet_name):
    if os.path.isfile(output_name):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)

def plot_task_means(df_all, output_dir):
    tasks = ["NC", "FB", "D1"]
    df_filtered = df_all[df_all["task"].isin(tasks)]
    df_filtered["task"] = df_filtered["task"].replace({"D1": "DB"})
    mean_values = df_filtered.groupby("task").mean()
    std_values = df_filtered.groupby("task").std()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 18

    for col in ["MeanVelocity_ax", "MeanVelocity_ay"]:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1,1,1)
        bar_width = 0.5
        index = np.arange(len(tasks))
        color_map = [cm.tab10(i) for i in range(len(index))]

        means = mean_values[col].reindex(["NC", "FB", "DB"])
        stds = std_values[col].reindex(["NC", "FB", "DB"])
        ax.bar(index, means, bar_width, yerr=stds, capsize=3, label=col, color=color_map)

        #ax.set_xlabel('Task')
        #ax.set_ylabel('Normalized Mean Velocity')
        ax.set_xticks(index)
        ax.set_xticklabels(["NC", "FB", "DB"])
        ax.set_ylim(0, 1.4)

        #plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"task_means_{col}.svg"))
        plt.close()

if __name__ == "__main__":
    main()
