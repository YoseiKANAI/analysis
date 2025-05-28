# %%
import os
import glob
import numpy as np
import pandas as pd
import global_value as g
from scipy.signal import butter, filtfilt
from scipy.stats import f
import matplotlib.pyplot as plt

class COPEllipseAreaCalculator:
    def __init__(self, data_dir, output_dir, subject_num):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.subject_num = subject_num
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_excel = os.path.join(self.output_dir, "result.xlsx")
        if os.path.isfile(self.output_excel):
            os.remove(self.output_excel)

    def run(self):
        all_rows = []
        for sub_id in range(1, self.subject_num + 1):
            sub_str = f"sub{sub_id:02d}"
            file_pattern = os.path.join(self.data_dir, f"{sub_str}*.csv")
            files = glob.glob(file_pattern)
            subject_rows = []
            for f in files:
                task = f[f.find(sub_str)+5:f.find(sub_str)+7]
                attempt = int(f[f.find(sub_str)+8:f.find(sub_str)+9])
                df = pd.read_csv(f)
                if ("ax" not in df.columns) or ("ay" not in df.columns):
                    continue
                # --- 前処理: ゼロ平均化 & 20Hzローパスフィルタ ---
                ax = df["ax"].values
                ay = df["ay"].values
                ax = ax - np.mean(ax)
                ay = ay - np.mean(ay)

                # ローパスフィルタ
                fs = 1000  # Hz
                cutoff = 20  # Hz
                b, a = butter(N=4, Wn=cutoff/(fs/2), btype='low')
                ax = filtfilt(b, a, ax)
                ay = filtfilt(b, a, ay)

                # --- 楕円面積計算 ---
                area = self.calc_ellipse_area(ax, ay)
                subject_rows.append({
                    "Subject": sub_str,
                    "Task": task,
                    "Attempt": attempt,
                    "EllipseArea": area
                })
            if subject_rows:
                df_sub = pd.DataFrame(subject_rows, columns=["Subject", "Task", "Attempt", "EllipseArea"])
                # Task列をカテゴリ型で順序付けし、g.task順でソート
                df_sub["Task"] = pd.Categorical(df_sub["Task"], categories=g.task, ordered=True)
                df_sub = df_sub.sort_values(["Task", "Attempt"]).reset_index(drop=True)
                # --- NC平均で正規化 ---
                nc_mean = df_sub[df_sub["Task"] == "NC"]["EllipseArea"].mean()
                df_sub["EllipseArea_norm"] = df_sub["EllipseArea"] / nc_mean if nc_mean != 0 else np.nan
                self.excel_output(df_sub, sub_str)
                all_rows.extend(df_sub.to_dict(orient="records"))
        # ALLシート
        if all_rows:
            df_all = pd.DataFrame(all_rows, columns=["Subject", "Task", "Attempt", "EllipseArea", "EllipseArea_norm"])
            df_all["Task"] = pd.Categorical(df_all["Task"], categories=g.task, ordered=True)
            df_all_sorted = df_all.sort_values(["Task", "Subject", "Attempt"]).reset_index(drop=True)
            self.excel_output(df_all_sorted, "ALL")
            # --- 箱ひげ図プロット ---
            self.plot_boxplot(
                df_all_sorted,
                value_col="EllipseArea_norm",
                output_dir=self.output_dir,
                filename="boxplot_EllipseArea_norm.png",
                ylabel="Ellipse Area (normalized)"
            )

    # 予測楕円面積計算
    def calc_ellipse_area(self, x, y):
        confidence = 0.95
        n = len(x)
        # 共分散行列を計算
        cov = np.cov(x, y)

        # 各軸の標準偏差を計算
        s_ml = np.sqrt(np.mean(x**2))
        s_ap = np.sqrt(np.mean(y**2))
        # 95%信頼区間の係数（フィッシャー分布：分子自由度2，分母自由度N-2の95%分位点）
        F = f.ppf(confidence, dfn=2, dfd=n-2)
        # 楕円面積を計算
        coeff =  ((n+1)*(n-1)) / (n*(n-2))
        det = np.sqrt( (s_ml**2) * (s_ap**2) - cov[0,1]**2 )
        area = 2 * np.pi * F * det * coeff
        return area

    def excel_output(self, df, sheet_name):
        if os.path.isfile(self.output_excel):
            with pd.ExcelWriter(self.output_excel, mode="a", if_sheet_exists="replace") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(self.output_excel) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def plot_boxplot(self, df, value_col, output_dir, filename,
                     task_order=None, task_labels=None, colors=None, ylim=[0,2.7], ylabel=None):
        """
        DataFrameからタスクごとに箱ひげ図を描画・保存する
        """
        import numpy as np
        import os

        # NC, FB, D1のみプロット
        plot_tasks = ["NC", "FB", "D1"]
        plot_labels = {"NC": "NC", "FB": "FB", "D1": "DBmass"}
        plot_colors = ["tab:blue", "tab:orange", "tab:green"]

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 24

        plt.figure(figsize=(8, 6))
        data = []
        means = []

        for t in plot_tasks:
            vals = df[df["Task"] == t][value_col].dropna().values
            vals = pd.to_numeric(vals, errors='coerce')
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)

        bplot = plt.boxplot(data, patch_artist=True, labels=[plot_labels[t] for t in plot_tasks],
                            medianprops=dict(color="black", linewidth=2))

        for patch, color in zip(bplot['boxes'], plot_colors):
            patch.set_facecolor(color)

        ax = plt.gca()
        for i, mean in enumerate(means):
            if not np.isnan(mean):
                ax.plot(i+1, mean, marker='x', color='black', markersize=10, zorder=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ylim is not None:
            plt.ylim(ylim)
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(value_col)
        plt.suptitle("")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()





if __name__ == "__main__":
    data_dir = rf"D:\user\kanai\Data\{g.datafile}\result_COP\dump"
    output_dir = rf"D:\user\kanai\Data\{g.datafile}\result_COP\Ellipse area"
    calculator = COPEllipseAreaCalculator(data_dir, output_dir, subject_num=g.subnum)
    calculator.run()
