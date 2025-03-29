# %%
# -*- coding: utf-8 -*-
# 生波形をMVCで正規化するコード
"""
Created on: 2025-01-04 19:34

@author: ShimaLab
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import detrend

import global_value as g

sampling = 2000

def main():
    for ID in range(g.subnum):
        # ルートフォルダのパスを指定
        root_dir = "D:/User/kanai/Data/%s/sub%d/csv/EMG_filterd/*.csv" %(g.datafile, ID+1)
        output_dir = "D:/User/kanai/Data/%s/sub%d/csv/RawMVC/" %(g.datafile, ID+1)
        
        result = pd.DataFrame(np.zeros((g.attempt,len(g.task))), columns=g.task)
        # 出力先フォルダを作成
        os.makedirs(output_dir, exist_ok=True)
        filename_list = glob.glob(root_dir)
        
        max_data = create_max_data(filename_list, ID)
        nomalization(filename_list ,max_data, output_dir)

###
### maxの格納した配列を作成
###
def create_max_data(filename_list, ID):
    result = pd.DataFrame(np.zeros((1, g.muscle_num)), columns=g.muscle_columns)
    
    # CSVファイルのパスを作成
    max_list = [s for s in filename_list if "M_" in s]    
    for f in max_list:
        df = pd.read_csv(f)
        df = df.abs()
        muscle = f[49:51]
        dirc = f[54]
        if dirc=="1":
            """
            if muscle == "SO" or muscle == "GM":
                result["GM_R"] = max(result["GM_R"].iloc[0], df["GM_R"].max())
                result["SO_R"] = max(result["SO_R"].iloc[0], df["SO_R"].max())
                result["PL_R"] = max(result["PL_R"].iloc[0], df["PL_R"].max())
                continue
            """
            if muscle == "GM":
                continue
            if muscle == "MF":
                result["MF_R"] = df["MF_R"].max()
                result["MF_L"] = df["MF_L"].max()
                continue
            if muscle == "SO":
                result["GM_R"] = df["GM_R"].max()
            if muscle == "GM":
                result["PL_R"] = df["PL_R"].max()
            columns = muscle + "_R"  
            result[columns] = df[columns].max()
            
        elif dirc=="2":
            """
            if muscle == "SO" or muscle == "GM":
                result["GM_L"] = max(result["GM_L"].iloc[0], df["GM_L"].max())
                result["SO_L"] = max(result["SO_L"].iloc[0], df["SO_L"].max())
                result["PL_L"] = max(result["PL_L"].iloc[0], df["PL_L"].max())
                continue
            """
            if muscle == "GM":
                continue
            if muscle == "SO":
                result["GM_L"] = df["GM_L"].max()
            if muscle == "TA":
                result["PL_L"] = df["PL_L"].max()
            columns = muscle + "_L"
            result[columns] = df[columns].max()
            
    return result

###
### maxで正規化
###
def nomalization(filename_list, max_data, output_dir):
    for path in filename_list:
        if "M_" in path:
            continue
        df = pd.read_csv(path)
        for i in g.muscle_columns:
            df[i] = df[i]/max_data[i].iloc[0]
        filename = path[47:]
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index = None)
        
if __name__ == "__main__":
    main()
# %%
