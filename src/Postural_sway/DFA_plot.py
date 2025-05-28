# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import global_value as g
import numpy as np

def main():
    # Excelファイルのパスを指定
    excel_path = f"D:/User/kanai/Data/{g.datafile}/result_COP/DFA/COP+obj/result.xlsx"

    # ExcelファイルからAllTasksシートを読み込み
    df = pd.read_excel(excel_path, sheet_name="AllTasks")

    # ボックスプロットを描画するカラム
    columns = ["COP_Alpha_x", "COP_Alpha_y", "Grasp_Alpha_X", "Grasp_Alpha_Y", "Grasp_Alpha_Z"]

    # 出力ディレクトリ
    output_dir = f"D:/User/kanai/Data/{g.datafile}/result_COP/DFA/COP+obj/plot"
    os.makedirs(output_dir, exist_ok=True)

    # プロット順とラベル・色の指定
    task_order = ["NC", "FB", "D1", "D2", "DW"]
    task_labels = {"NC": "NC", "FB": "FB", "D1": "DBmass", "D2": "DBchar", "DW": "DW"}
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]  # tab:20から5色

    # Task列をカテゴリ型で順序付け
    df["Task"] = pd.Categorical(df["Task"], categories=task_order, ordered=True)

    # 被検者ごとのシート名リスト
    subject_sheets = [f"sub{i+1}" for i in range(g.subnum)]

    # 全体データ（AllTasksシート）でのボックスプロット
    for col in columns:
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 18

        plt.figure(figsize=(6, 4))

        # Task列の値でタスクを区別して抽出
        if col.startswith("Grasp_Alpha"):
            plot_tasks = ["FB", "D1", "D2"]
            plot_labels = [task_labels[t] for t in plot_tasks]
            plot_colors = colors[1:4]
        else:
            plot_tasks = ["NC", "FB", "D1"]
            plot_labels = [task_labels[t] for t in plot_tasks]
            plot_colors = colors[0:3]

        # データを抽出し、indexをリセットして値だけを使う
        data = []
        means = []
        for t in plot_tasks:
            vals = df[df["Task"] == t][col].dropna().values
            vals = pd.to_numeric(vals, errors='coerce')
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)

        bplot = plt.boxplot(data, patch_artist=True, labels=plot_labels,
                            medianprops=dict(color="black", linewidth=2))

        # 色付け
        for patch, color in zip(bplot['boxes'], plot_colors):
            patch.set_facecolor(color)

        # 平均値をマーク（×）で追加
        ax = plt.gca()
        for i, mean in enumerate(means):
            if not np.isnan(mean):
                ax.plot(i+1, mean, marker='x', color='black', markersize=10, zorder=3)

        # 枠線の調整（上・右を消す）
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # y軸範囲の設定
        if col.startswith("Grasp_Alpha"):
            plt.ylim(bottom=0.0, top=2.0)
        else:
            plt.ylim(bottom=0.6, top=1.5)

        #plt.title(f"Boxplot of {col} by Task")
        plt.suptitle("")
        #plt.xlabel("Task")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
        plt.close()

    # 被検者ごとのデータでもボックスプロットを出力
    for sub_sheet in subject_sheets:
        try:
            df_sub = pd.read_excel(excel_path, sheet_name=sub_sheet)
        except Exception:
            continue  # シートがなければスキップ

        df_sub["Task"] = pd.Categorical(df_sub["Task"], categories=task_order, ordered=True)

        for col in columns:
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams["font.size"] = 18

            plt.figure(figsize=(6, 4))

            if col.startswith("Grasp_Alpha"):
                plot_tasks = ["FB", "D1", "D2"]
                plot_labels = [task_labels[t] for t in plot_tasks]
                plot_colors = colors[1:4]
            else:
                plot_tasks = ["NC", "FB", "D1"]
                plot_labels = [task_labels[t] for t in plot_tasks]
                plot_colors = colors[0:3]

            data = []
            means = []
            for t in plot_tasks:
                vals = df_sub[df_sub["Task"] == t][col].dropna().values
                vals = pd.to_numeric(vals, errors='coerce')
                vals = vals[np.isfinite(vals)]
                data.append(vals)
                means.append(np.mean(vals) if len(vals) > 0 else np.nan)

            bplot = plt.boxplot(data, patch_artist=True, labels=plot_labels,
                                medianprops=dict(color="black", linewidth=2))

            for patch, color in zip(bplot['boxes'], plot_colors):
                patch.set_facecolor(color)

            ax = plt.gca()
            for i, mean in enumerate(means):
                if not np.isnan(mean):
                    ax.plot(i+1, mean, marker='x', color='black', markersize=10, zorder=3)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if col.startswith("Grasp_Alpha"):
                plt.ylim(bottom=0.0, top=2.0)
            else:
                plt.ylim(bottom=0.6, top=1.5)

            plt.suptitle("")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sub_sheet}_boxplot_{col}.png"))
            plt.close()

    # 被検者ごとのデータを横並びで一つの画像に出力
    subject_sheets = [f"sub{i+1}" for i in range(g.subnum)]
    for col in columns:
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 18

        n_tasks = 3 if not col.startswith("Grasp_Alpha") else 3  # NC, FB, D1 or FB, D1, D2
        n_subjects = len(subject_sheets)
        box_width = 0.7
        group_gap = 0.7  # 被験者間の隙間を小さく
        task_gap = 0.0   # タスク間の隙間（密着）

        xtick_pos = []
        xtick_labels = []
        all_data = []
        all_colors = []
        subject_centers = []
        pos = 1

        for sub_idx, sub_sheet in enumerate(subject_sheets):
            try:
                df_sub = pd.read_excel(excel_path, sheet_name=sub_sheet)
            except Exception:
                continue  # シートがなければスキップ

            df_sub["Task"] = pd.Categorical(df_sub["Task"], categories=task_order, ordered=True)

            if col.startswith("Grasp_Alpha"):
                plot_tasks = ["FB", "D1", "D2"]
                plot_colors = colors[1:4]
            else:
                plot_tasks = ["NC", "FB", "D1"]
                plot_colors = colors[0:3]

            start_pos = pos
            for i, t in enumerate(plot_tasks):
                vals = df_sub[df_sub["Task"] == t][col].dropna().values
                vals = pd.to_numeric(vals, errors='coerce')
                vals = vals[np.isfinite(vals)]
                all_data.append(vals)
                all_colors.append(plot_colors[i])
                xtick_pos.append(pos)
                pos += 1  # タスクごとに+1
            # 被検者の中央位置を計算しラベル用に保存
            subject_centers.append((start_pos + pos - 1) / 2)
            xtick_labels.append(sub_sheet)
            pos += group_gap  # 被験者ごとに隙間

        fig = plt.figure(figsize=(10, 6))  # 横圧縮・縦拡大
        ax = fig.add_subplot(1, 1, 1)

        bplot = ax.boxplot(all_data, patch_artist=True, positions=xtick_pos,
                           widths=box_width, medianprops=dict(color="black", linewidth=2))

        for patch, color in zip(bplot['boxes'], all_colors):
            patch.set_facecolor(color)

        # 平均値を×で追加
        for i, vals in enumerate(all_data):
            if len(vals) > 0:
                ax.plot(xtick_pos[i], np.mean(vals), marker='x', color='black', markersize=10, zorder=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if col.startswith("Grasp_Alpha"):
            plt.ylim(bottom=0.0, top=2.0)
        else:
            plt.ylim(bottom=0.6, top=1.5)

        plt.suptitle("")
        plt.ylabel(col)
        # x軸ラベルは被検者名のみ
        ax.set_xticks(subject_centers)
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"all_subjects_boxplot_{col}.png"))
        plt.close()

if __name__ == "__main__":
    main()
