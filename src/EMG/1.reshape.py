# %%
# -*- coding: utf-8 -*-
# EMGのチャンネルを統一するため，指定したファイルのカラムを変更する

"""
Created on: 2024-09-22 15:59

@author: ShimaLab
"""
import os
import pandas as pd
import global_value as g

for i in range(g.subnum):
    ID = i+1
    # ルートフォルダのパスを指定
    root_dir = "D:/User/kanai/Data/%s/sub%d/csv/" %(g.datafile, ID)

    # ルートフォルダ以下のすべてのフォルダに対して処理を実行
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # フォルダ内のすべてのcsvファイルに対して処理を実行
        for filename in filenames:
            if filename.endswith(("_a_2.csv")):
                # CSVファイルのパスを作成
                input_path = os.path.join(dirpath, filename)
                df = pd.read_csv(input_path)

                if ID==4 or ID==5:
                    # カラム名を変更する
                    df = df.rename(columns={'CH9': 'SO_R'})# ヒラメ筋
                    df = df.rename(columns={'CH10': 'SO_L'})
                    df = df.rename(columns={'CH11': 'GM_R'})# 腓腹筋
                    df = df.rename(columns={'CH12': 'GM_L'})
                    df = df.rename(columns={'CH13': 'TA_R'})# 前脛骨筋
                    df = df.rename(columns={'CH5': 'TA_L'})
                    df = df.rename(columns={'CH15': 'PL_R'})# 長腓骨筋
                    df = df.rename(columns={'CH16': 'PL_L'})
                    df = df.rename(columns={'CH1': 'IO_R'})# 内腹斜筋
                    df = df.rename(columns={'CH2': 'IO_L'})
                    df = df.rename(columns={'CH3': 'MF_R'})# 多裂筋
                    df = df.rename(columns={'CH4': 'MF_L'})

                elif ID==9:
                    # カラム名を変更する
                    df = df.rename(columns={'CH9': 'SO_R'})# ヒラメ筋
                    df = df.rename(columns={'CH10': 'SO_L'})
                    df = df.rename(columns={'CH11': 'GM_R'})# 腓腹筋
                    df = df.rename(columns={'CH12': 'GM_L'})
                    df = df.rename(columns={'CH13': 'TA_R'})# 前脛骨筋
                    df = df.rename(columns={'CH14': 'TA_L'})
                    df = df.rename(columns={'CH15': 'PL_R'})# 長腓骨筋
                    df = df.rename(columns={'CH16': 'PL_L'})
                    df = df.rename(columns={'CH1': 'IO_R'})# 内腹斜筋
                    df = df.rename(columns={'CH5': 'IO_L'})
                    df = df.rename(columns={'CH3': 'MF_R'})# 多裂筋
                    df = df.rename(columns={'CH7': 'MF_L'})

                else:
                    # カラム名を変更する
                    df = df.rename(columns={'CH9': 'SO_R'})# ヒラメ筋
                    df = df.rename(columns={'CH10': 'SO_L'})
                    df = df.rename(columns={'CH11': 'GM_R'})# 腓腹筋
                    df = df.rename(columns={'CH12': 'GM_L'})
                    df = df.rename(columns={'CH13': 'TA_R'})# 前脛骨筋
                    df = df.rename(columns={'CH14': 'TA_L'})
                    df = df.rename(columns={'CH15': 'PL_R'})# 長腓骨筋
                    df = df.rename(columns={'CH16': 'PL_L'})
                    df = df.rename(columns={'CH1': 'IO_R'})# 内腹斜筋
                    df = df.rename(columns={'CH2': 'IO_L'})
                    df = df.rename(columns={'CH3': 'MF_R'})# 多裂筋
                    df = df.rename(columns={'CH4': 'MF_L'})
                df.to_csv(input_path, index=None)
        break
# %%
