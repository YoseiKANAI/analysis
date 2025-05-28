# %%
# -*- coding: utf-8 -*-
# COPとForceの相互相関解析を行う
"""
Created on Wed May  3 18:41:24 2023

@author: ShimaLab
"""

# "C:/Users/ShimaLab/Desktop/one time/result1"
# "C:/Users/ShimaLab/Desktop/one time/result2"
# "C:/Users/ShimaLab/Desktop/one time/output"
import os
import csv
import numpy as np
import global_value as g
from glob import glob  # globをインポート
import pandas as pd  # pandasをインポート

#TODO: この関数使われていないから要チェック
def compute_correlation_and_write(writer, F, COP, F_file, type, mode):
    for j in range(3):
        max_correlation = float('-inf')
        min_lag = None
        for k in range(-50, 51):
            correlation = np.corrcoef(F[:, mode], np.roll(COP[:, j], k))[0, 1]
            if correlation > max_correlation:
                max_correlation = correlation
                min_lag = k
        lag = min_lag * 10
        if j == 0:
            writer.writerow([F_file, type[j], lag, max_correlation])
        else:
            writer.writerow(["", type[j], lag, max_correlation])


def read_combined_data(file_path):
    """
    単一のCSVファイルからForce, COP, COP_velocityデータを分割して読み取る
    """
    df = pd.read_csv(file_path)
    force_data = df[["Force X", "Force Y", "Force Z"]]
    cop_data = df[["COP_X", "COP_Y", "COP"]]
    cop_velocity_data = df[["COP_velocity_X", "COP_velocity_Y", "COP_velocity"]]
    return force_data.iloc[:2970, :], cop_data.iloc[:2970, :], cop_velocity_data.iloc[:2970, :]


def calculate_correlation(input_dir, ID):
    # タスクと試行ごとの結果を保持するための辞書
    results_dict = {}

    files = sorted(glob(input_dir + "*.csv"))

    for file_path in files:
        # ファイル名からタスクと試行番号を抽出
        file_name = os.path.basename(file_path)
        task = file_name[0:2]  # 例: "D1"
        attempt = int(file_name[2:4])  # 例: "02"

        # データを読み取る
        force, cop, cop_velocity = read_combined_data(file_path)

        # キーを作成 (task-attempt)
        key = f"{task}-{attempt}"

        # このファイルの結果を格納する辞書を初期化
        if key not in results_dict:
            results_dict[key] = {"task": task, "subject": ID + 1, "attempt": attempt}

        # 相関係数とラグを計算
        for force_axis in ["Force X", "Force Y", ]:  # Force X, Y
            for cop_type, df_cop in [("COP", cop), ("COP_velocity", cop_velocity)]:
                for cop_axis in ["_X", "_Y", ""]:  # COP_X, COP_Y or COP_velocity_X, COP_velocity_Y
                    cop_columns = cop_type + cop_axis  # COP_X, COP_Y, COP_velocity_X, COP_velocity_Y
                    max_correlation = float('-inf')
                    min_lag = 0
                    for k in range(0, 51):
                        correlation = np.corrcoef(force[force_axis], np.roll(df_cop[cop_columns], k))[0, 1]
                        if correlation > max_correlation:
                            max_correlation = correlation
                            min_lag = k
                    lag = min_lag * 10

                    # カラム名を生成
                    col_r = f"r-{force_axis}-{cop_columns}"
                    col_lag = f"Lag-{force_axis}-{cop_columns}"

                    # 相関係数とラグを結果辞書に追加
                    results_dict[key][col_r] = max_correlation
                    results_dict[key][col_lag] = lag

    # 辞書をリストに変換
    results = list(results_dict.values())
    return results


def organize_data_for_excel(results):
    """
    結果データをExcel出力用に整理する
    """
    if not results:
        return pd.DataFrame()

    # すべての行から列名を取得
    all_columns = set()
    for row in results:
        all_columns.update(row.keys())

    # 'task', 'subject', 'attempt'を除く列名を取得
    data_columns = [col for col in all_columns if col not in ['task', 'subject', 'attempt']]

    # 相関係数の列とラグの列に分ける
    r_columns = sorted([col for col in data_columns if col.startswith('r-')])
    lag_columns = sorted([col for col in data_columns if col.startswith('Lag-')])

    # 整理されたデータを作成
    organized_data = []
    for row in results:
        new_row = {'task': row['task'], 'subject': row['subject'], 'attempt': row['attempt']}

        # 相関係数の列を追加
        for col in r_columns:
            new_row[col] = row.get(col, None)

        # 空白列を追加
        new_row[''] = ''

        # ラグの列を追加
        for col in lag_columns:
            new_row[col] = row.get(col, None)

        organized_data.append(new_row)

    return pd.DataFrame(organized_data)


def write_to_excel(all_results, output_dir):
    output_name = f"{output_dir}/Force_COP_Correlation.xlsx"
    with pd.ExcelWriter(output_name) as writer:
        # 各被験者のデータを個別シートに出力
        for ID, results in enumerate(all_results, start=1):
            # データを整理
            df_results = organize_data_for_excel(results)
            sheet_name = f"sub{ID}"
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)

        # すべてのデータをまとめたシートを作成
        all_data = pd.concat([organize_data_for_excel(results) for results in all_results], ignore_index=True)
        # タスク、被験者ID、試行番号でソート
        all_data = all_data.sort_values(by=["task", "subject", "attempt"]).reset_index(drop=True)
        all_data.to_excel(writer, sheet_name="AllData", index=False)


def main():
    # メイン処理
    all_results = []
    for ID in range(g.subnum):
        input_dir = f"D:/User/kanai/Data/{g.datafile}/sub{ID+1}/csv/Force_COP/"
        output_dir = f"D:/User/kanai/Data/{g.datafile}/result_CAA/COP_Force/"
        os.makedirs(output_dir, exist_ok=True)
        results = calculate_correlation(input_dir, ID)
        all_results.append(results)

    # すべてのデータを1つのExcelファイルに出力
    write_to_excel(all_results, output_dir)

if __name__ == "__main__":
    main()