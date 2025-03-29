# %%
# -*- coding: utf-8 -*-
# MVCでバランスタスクを正規化

"""
Created on: 2024-12-16 21:25

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np

import global_value as g

sampling = 2000

def main():
    for ID in range(g.subnum):
        file_list, output_dir = preparation(ID)
        
        max_data = pd.DataFrame(np.zeros((1, g.muscle_num)), columns=g.muscle_columns)
        create_max_data(max_data, file_list)
        nomalization(max_data, file_list,output_dir)

"""
筋電データののみのパスリストを作成
"""
def preparation(ID):
    # ルートフォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/sub%d/csv/EMG_proc/*.csv" %(g.datafile, ID+1)
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/MVC" %(g.datafile, ID+1)
    
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    file_list = glob.glob(input_dir)
    
    return file_list, output_dir

"""
maxを格納した配列を作成
"""
def create_max_data(max_data, file_list):
    # CSVファイルのパスを作成
    max_list = [s for s in file_list if "M_" in s]    
    for path in max_list:
        df = pd.read_csv(path)
        muscle = path[path.find("\\")+3 : path.find(".")]
        
        # 筋ごとにmaxを格納
        # ヒラメ筋，腓腹筋，長腓骨筋
        if "SO" in muscle:
            # ヒラメ筋
            max_data[muscle] = max(max_data[muscle].iloc[0], df[muscle].max())
            # 腓腹筋
            GM = muscle.replace("SO", "GM")
            max_data[GM] = max(max_data[GM].iloc[0], df[GM].max())
            # 長腓骨筋
            PL = muscle.replace("SO", "PL")
            max_data[PL] = max(max_data[PL].iloc[0], df[PL].max())
      
        # 前脛骨筋
        elif "TA" in muscle:
            max_data[muscle] = max(max_data[muscle].iloc[0], df[muscle].max())
        
        # 体幹筋
        else:
            R = muscle + "_R"
            L = muscle + "_L"
            max_data[R] = max(max_data[R].iloc[0], df[R].max())
            max_data[L] = max(max_data[L].iloc[0], df[L].max())

"""
maxで正規化
"""
def nomalization(max_data, file_list, output_dir):
    for path in file_list:
        if "M_" in path:
            continue
        df = pd.read_csv(path)
        for i in g.muscle_columns:
            df[i] = df[i]/max_data[i].iloc[0]
        filename = path[path.find("\\")+1:]
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index = None)
            
if __name__ == "__main__":
    main()
# %%
