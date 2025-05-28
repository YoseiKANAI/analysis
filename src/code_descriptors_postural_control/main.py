# %%
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import global_value as g
import glob
import re
from scipy.signal import decimate  # 追加

from stabilogram.stato import Stabilogram
from descriptors import compute_all_features

class COPFeatureCalculator:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_excel = os.path.join(self.output_dir, "result_features.xlsx")
        self.file_list = glob.glob(os.path.join(self.data_dir, "*.csv"))

    def run(self):
        all_rows = []
        for f in self.file_list:
            # ファイル名から被検者・タスク・試行番号抽出
            basename = os.path.basename(f)
            m = re.match(r"sub(\d{2})([A-Z0-9]{2})(\d{2})\.csv", basename)
            if not m:
                continue
            subject = f"sub{m.group(1)}"
            task = m.group(2)
            attempt = int(m.group(3))
            if task not in g.task:
                continue

            # データ読み込み
            df = pd.read_csv(f)
            # NaNを線形補間
            df["ax"] = df["ax"].interpolate(method="linear", limit_direction="both")
            df["ay"] = df["ay"].interpolate(method="linear", limit_direction="both")

            # 1000Hz→100Hzにダウンサンプリング
            #ax_ds = decimate(df["ax"].values, 10, ftype='fir', zero_phase=True)
            #ay_ds = decimate(df["ay"].values, 10, ftype='fir', zero_phase=True)

            # COP計算
            X = df["ax"] - np.mean(df["ax"])
            Y = df["ay"] - np.mean(df["ay"])

            # 単位をcmに変換
            X = X/10
            Y = Y/10

            data = np.array([X, Y]).T
            stato = Stabilogram()
            stato.from_array(array=data, original_frequency=1000)
            sway_density_radius = 0.3 # 3 mm

            params_dic = {"sway_density_radius": sway_density_radius}
            # 特徴量計算
            features = compute_all_features(stato, params_dic=params_dic)


            row = {
                "Subject": subject,
                "Task": task,
                "Attempt": attempt,
            }
            row.update(features)
            all_rows.append(row)

        # DataFrame化・Excel出力
        if all_rows:
            df_all = pd.DataFrame(all_rows)
            df_all["Task"] = pd.Categorical(df_all["Task"], categories=g.task, ordered=True)
            # ソート順を Task, Subject, Attempt の順に変更
            df_all = df_all.sort_values(["Task", "Subject", "Attempt"]).reset_index(drop=True)
            self.excel_output(df_all)

    def excel_output(self, df_all):
        with pd.ExcelWriter(self.output_excel) as writer:
            for subject, df_sub in df_all.groupby("Subject"):
                df_sub = df_sub.sort_values(["Task", "Attempt"]).reset_index(drop=True)
                df_sub.to_excel(writer, sheet_name=subject, index=False)
            df_all.to_excel(writer, sheet_name="ALL", index=False)

if __name__ == "__main__":
    data_dir = rf"D:\user\kanai\Data\{g.datafile}\result_COP\dump"
    output_dir = rf"D:\user\kanai\Data\{g.datafile}\result_COP\opencode"
    calculator = COPFeatureCalculator(data_dir, output_dir)
    calculator.run()

