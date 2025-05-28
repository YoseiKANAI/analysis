# %%
# -*- coding: utf-8 -*-
# 相互相関用にデータを作成
"""
Created on Thu Apr 27 14:55:23 2023

@author: ShimaLab
"""

import os
import math
import pandas as pd
import numpy as np
import scipy.signal
import glob
import global_value as g
from scipy.signal import butter, filtfilt, decimate

# 重力加速度 [m/s^2]
gravity = 9.806650
# サビツキーゴーレイフィルタのパラメータ
window_length = 71  # ウィンドウサイズ（奇数）
order = 3          # 多項式の次数

def output_prepare(root_dir):
    """出力先フォルダを作成"""
    output_dir = os.path.join(root_dir, "Force")
    os.makedirs(output_dir, exist_ok=True)

    output_dir_COP = os.path.join(root_dir, "COP_Standard")
    os.makedirs(output_dir_COP, exist_ok=True)
    return output_dir, output_dir_COP


def calculate_force(df, filename):
    """Forceデータを計算"""
    # 指先データ
    finger = pd.DataFrame()
    finger["X"] = df["finger X"]
    finger["Y"] = df["finger Y"]
    finger["Z"] = df["finger Z"]


    # 基準データ
    base = pd.DataFrame()
    if filename.startswith(("DW")):
        base["X"] = df["weight X"]
        base["Y"] = df["weight Y"]
        base["Z"] = df["weight Z"]
    else:
        base["X"] = df["base X"]
        base["Y"] = df["base Y"]
        base["Z"] = df["base Z"]

    # 相対加速度を算出する
    relative_pos = pd.DataFrame()
    relative_pos = base - finger

    # DataFrameに変換
    finger.columns = ["finger_X", "finger_Y", "finger_Z"]
    base.columns = ["base_X", "base_Y", "base_Z"]
    relative_pos.columns = ["RP_X", "RP_Y", "RP_Z"]

    # 相対速度を算出
    relative_ve = pd.DataFrame()
    relative_ve["RV_X"] = relative_pos["RP_X"].diff().shift(-1) / 0.01
    relative_ve["RV_Y"] = relative_pos["RP_Y"].diff().shift(-1) / 0.01
    relative_ve["RV_Z"] = relative_pos["RP_Z"].diff().shift(-1) / 0.01

    # 相対速度を算出
    relative_ac = pd.DataFrame()
    relative_ac["RA_X"] = relative_ve["RV_X"].diff().shift(-1) / 0.01
    relative_ac["RA_Y"] = relative_ve["RV_Y"].diff().shift(-1) / 0.01
    relative_ac["RA_Z"] = relative_ve["RV_Z"].diff().shift(-1) / 0.01

    # 相対加速度を基に力を算出
    task_name = filename[0:2]
    Force = relative_ac * g.m[task_name] * 10**(-3)
    Force.columns = ["Force_X", "Force_Y", "Force_Z"]
    if filename.startswith("F"):
        Force.iloc[:, 2] = Force.iloc[:, 2] - g.m[task_name] * gravity * 10**(-3)
    else:
        Force.iloc[:, 2] = Force.iloc[:, 2] + g.m[task_name] * gravity * 10**(-3)

    for axis in ["X", "Y", "Z"]:
        # サビツキーゴーレイフィルタ適用
        Force["Force_" + axis] = scipy.signal.savgol_filter(Force["Force_" + axis], window_length, order)
        #relative_pos["RP_" + axis] = scipy.signal.savgol_filter(relative_pos["RP_" + axis], window_length, order)
        relative_ve["RV_" + axis] = scipy.signal.savgol_filter(relative_ve["RV_" + axis], window_length, order)
        relative_ac["RA_" + axis] = scipy.signal.savgol_filter(relative_ac["RA_" + axis], window_length, order)

        # 分散1、平均0で標準化
        Force["Force_" + axis] = (Force["Force_" + axis] - Force["Force_" + axis].mean()) / Force["Force_" + axis].std()
        #relative_pos["RP_" + axis] = (relative_pos["RP_" + axis] - relative_pos["RP_" + axis].mean()) / relative_pos["RP_" + axis].std()
        relative_ve["RV_" + axis] = (relative_ve["RV_" + axis] - relative_ve["RV_" + axis].mean()) / relative_ve["RV_" + axis].std()
        relative_ac["RA_" + axis] = (relative_ac["RA_" + axis] - relative_ac["RA_" + axis].mean()) / relative_ac["RA_" + axis].std()

    return pd.concat([finger, base, relative_pos, relative_ve, relative_ac, Force], axis=1)


def apply_lowpass_filter(data, cutoff=10, fs=1000, order=4):
    """50Hzローパスフィルタを適用"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_cop(cop_file):
    """COPデータを計算"""
    cop_df = pd.read_csv(cop_file, names=("COP_X", "COP_Y", "COP"), skiprows=1)
    for j in range(len(cop_df)):
        cop_df.iloc[j, 2] = math.sqrt(cop_df.iloc[j, 0]**2 + cop_df.iloc[j, 1]**2)

    # 50Hzローパスフィルタ適用
    for col in ["COP_X", "COP_Y", "COP"]:
        cop_df[col] = apply_lowpass_filter(cop_df[col])

    cop_df_resample = pd.DataFrame()
    # 1000Hzから100Hzへダウンサンプリング
    for col in ["COP_X", "COP_Y", "COP"]:
        cop_df_resample[col] = decimate(cop_df[col], q=10, zero_phase=True)

    # COP速度を算出
    cop_velo = pd.DataFrame()
    cop_velo["COP_X"] = cop_df_resample["COP_X"].diff().shift(-1) / 0.01
    cop_velo["COP_Y"] = cop_df_resample["COP_Y"].diff().shift(-1) / 0.01
    cop_velo["COP"] = cop_df_resample["COP"].diff().shift(-1) / 0.01

    # COP加速度を算出
    cop_acc = pd.DataFrame()
    cop_acc["COP_X"] = cop_velo["COP_X"].diff().shift(-1) / 0.01
    cop_acc["COP_Y"] = cop_velo["COP_Y"].diff().shift(-1) / 0.01
    cop_acc["COP"] = cop_velo["COP"].diff().shift(-1) / 0.01

    # サビツキーゴーレイフィルタ適用
    for col in ["COP_X", "COP_Y", "COP"]:
        cop_df_resample[col] = scipy.signal.savgol_filter(cop_df_resample[col], window_length, order)
        cop_velo[col] = scipy.signal.savgol_filter(cop_velo[col], window_length, order)
        cop_acc[col] = scipy.signal.savgol_filter(cop_acc[col], window_length, order)

    # 分散1、平均0で標準化
    for col in ["COP_X", "COP_Y", "COP"]:
        cop_df_resample[col] = (cop_df_resample[col] - cop_df_resample[col].mean()) / cop_df_resample[col].std()
        cop_velo[col] = (cop_velo[col] - cop_velo[col].mean()) / cop_velo[col].std()
        cop_acc[col] = (cop_acc[col] - cop_acc[col].mean()) / cop_acc[col].std()

    cop_velo.columns = ["COP_velo_X", "COP_velo_Y", "COP_velo"]
    cop_acc.columns = ["COP_acc_X", "COP_acc_Y", "COP_acc"]
    # COPデータ、速度、加速度を統合
    return pd.concat([cop_df_resample, cop_velo, cop_acc], axis=1)


def extract_center_of_gravity(file_path, task_name, base=None):
    """物体の重心座標を抽出し、速度と加速度を算出"""
    if task_name == "DW" and base is not None:
        # DWタスクの場合、baseデータをobjデータとして使用
        obj_data = base.rename(columns={"base_X": "obj_X", "base_Y": "obj_Y", "base_Z": "obj_Z"})
    else:
        df = pd.read_csv(file_path)
        obj_columns = [col for col in df.columns if task_name in col and col.endswith("X")]
        if not obj_columns:
            return pd.DataFrame()
        obj_x_col = obj_columns[0]
        obj_y_col = df.columns[df.columns.get_loc(obj_x_col) + 1]
        obj_z_col = df.columns[df.columns.get_loc(obj_x_col) + 2]
        obj_data = df[[obj_x_col, obj_y_col, obj_z_col]].copy()
        obj_data.columns = ["obj_X", "obj_Y", "obj_Z"]

    # 重心座標の速度を算出
    obj_velo = pd.DataFrame()
    obj_velo["obj_velo_X"] = obj_data["obj_X"].diff().shift(-1) / 0.01
    obj_velo["obj_velo_Y"] = obj_data["obj_Y"].diff().shift(-1) / 0.01
    obj_velo["obj_velo_Z"] = obj_data["obj_Z"].diff().shift(-1) / 0.01

    # 重心座標の加速度を算出
    obj_acc = pd.DataFrame()
    obj_acc["obj_acc_X"] = obj_velo["obj_velo_X"].diff().shift(-1) / 0.01
    obj_acc["obj_acc_Y"] = obj_velo["obj_velo_Y"].diff().shift(-1) / 0.01
    obj_acc["obj_acc_Z"] = obj_velo["obj_velo_Z"].diff().shift(-1) / 0.01

    # サビツキーゴーレイフィルタ適用
    for col in ["obj_X", "obj_Y", "obj_Z"]:
        obj_data[col] = scipy.signal.savgol_filter(obj_data[col], window_length, order)
    for col in ["obj_velo_X", "obj_velo_Y", "obj_velo_Z"]:
        obj_velo[col] = scipy.signal.savgol_filter(obj_velo[col], window_length, order)
    for col in ["obj_acc_X", "obj_acc_Y", "obj_acc_Z"]:
        obj_acc[col] = scipy.signal.savgol_filter(obj_acc[col], window_length, order)

    # 分散1、平均0で標準化
    for col in ["obj_X", "obj_Y", "obj_Z"]:
        obj_data[col] = (obj_data[col] - obj_data[col].mean()) / obj_data[col].std()
    for col in ["obj_velo_X", "obj_velo_Y", "obj_velo_Z"]:
        obj_velo[col] = (obj_velo[col] - obj_velo[col].mean()) / obj_velo[col].std()
    for col in ["obj_acc_X", "obj_acc_Y", "obj_acc_Z"]:
        obj_acc[col] = (obj_acc[col] - obj_acc[col].mean()) / obj_acc[col].std()

    return pd.concat([obj_data, obj_velo, obj_acc], axis=1)


def process_and_calculate_force_and_cop(root_dir, input_COP, output_dir, ID):
    """力覚データ、COPデータ、重心座標を処理し、同じファイルに格納して出力フォルダに保存"""
    csv_files = glob.glob(os.path.join(root_dir, "*.csv"), recursive=True)
    cop_files = glob.glob(os.path.join(input_COP, "*.csv"), recursive=True)

    for input_path in csv_files:
        filename = os.path.basename(input_path)
        if filename.endswith(("f_1.csv", "f_2.csv", "_2D.csv", "_6D.csv", "_a.csv", "_a_1.csv", "_a_2.csv")):
            continue

        subject_id = "sub%02d" % (ID + 1)  # 被検者番号
        task_name = filename[:2]          # タスク名
        trial_number = filename[-6:-4]    # 試行番号（例: "D10002.csv" の "02"）

        output_filename = filename.replace("00", "")
        output_path = os.path.join(output_dir, output_filename)

        if filename.startswith(("NC")):
            # NCはCOPデータのみ抽出
            cop_file = next((f for f in cop_files if f"{subject_id}{task_name}{trial_number}" in os.path.basename(f)), None)
            if cop_file:
                cop_data = calculate_cop(cop_file)
                cop_data.to_csv(output_path, index=False)
            continue

        df = pd.read_csv(input_path)
        if "finger X" not in df.columns:
            continue

        # Forceデータを計算
        force_data = calculate_force(df, filename)

        # 対応するCOPデータを検索して計算
        cop_file = next((f for f in cop_files if f"{subject_id}{task_name}{trial_number}" in os.path.basename(f)), None)
        if cop_file:
            cop_data = calculate_cop(cop_file)

            # 対応するobjデータを検索して抽出
            obj_file = next((f for f in csv_files if f"{task_name}00{trial_number}_6D.csv" in os.path.basename(f)), None)
            obj_data = extract_center_of_gravity(obj_file, task_name, base=force_data[["base_X", "base_Y", "base_Z"]] if task_name == "DW" else None)

            # Force、COP、objを統合して出力
            result = pd.concat([force_data, cop_data, obj_data], axis=1)
            result.to_csv(output_path, index=False)


def main():
    """メイン処理"""
    for ID in range(g.subnum):
        root_dir = "D:/User/kanai/Data/%s/sub%d/csv" % (g.datafile, ID + 1)
        input_COP = "D:/User/kanai/Data/%s/result_COP/dump" % (g.datafile)
        output_dir = os.path.join(root_dir, "Force_COP")
        os.makedirs(output_dir, exist_ok=True)

        process_and_calculate_force_and_cop(root_dir, input_COP, output_dir, ID)


if __name__ == "__main__":
    main()