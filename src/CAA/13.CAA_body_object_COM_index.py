# %%
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:57:50 2023

@author: ShimaLab
"""
# input_folder = "C:/Users/ShimaLab/Desktop/one time"
# output_folder = "C:/Users/ShimaLab/Desktop/one time/out"
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_value as g

task = ["DB", "W"]
index = ["X-X", "X-Y", "X-COM", "Y-X", "Y-Y", "Y-COM",]
color = ["tab:green", "tab:red"]
task_num = 2
for ID in range(g.subnum):
    subID = "sub%d" %(ID+1)
    # ルートフォルダのパスを指定
    input_dir = "D:/User/kanai/Data/%s/result_CAA/obj-COM/sub%d" %(g.datafile, ID+1)    
    output_dir = "D:/User/kanai/Data/%s/result_CAA/index_plot" %(g.datafile) 
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.startswith("sub"):
                # CSVファイルを読み込みます。
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path)
    
    index_num = np.arange(6)
    corr = pd.DataFrame([df.iloc[1, :], df.iloc[5, :]], index=[0, 1])            
    # グラフをプロット
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15   

    # 相関係数
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(1,1,1)
    
    for t in range(2):
        slide = t*0.4
        err = [df.iloc[t*4+3, 0], df.iloc[t*4+3, 1], df.iloc[t*4+3, 2], df.iloc[t*4+3, 3], df.iloc[t*4+3, 4], df.iloc[t*4+3, 5]]
        ax.bar(index_num+slide, corr.iloc[t, :], width=0.3, yerr=err, capsize=3, label = task[t], color = color[t])
    ax.legend(loc = "upper right", fontsize ="large", ncol=len(task), frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.25)
    ax.set_ylim([0.0, 0.4])
    ax.set_xlabel("Object-COM relation")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_xticklabels(index)

    plt.show()
    fig.savefig(output_dir + "/plot_" + subID + "_corr.png")
    
        
    # ラグ
    index_num = np.arange(6)
    lag = pd.DataFrame([df.iloc[0, :], df.iloc[4, :]], index=[0, 1])            
    # グラフをプロット
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15   

    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(1,1,1)
    
    for t in range(2):
        slide = t*0.46
        err = [df.iloc[t*4+2, 0], df.iloc[t*4+2, 1], df.iloc[t*4+2, 2], df.iloc[t*4+2, 3], df.iloc[t*4+2, 4], df.iloc[t*4+2, 5]]
        ax.bar(index_num+slide, lag.iloc[t, :], width=0.3, capsize=3, yerr = err, label = task[t], color = color[t])
    ax.legend(loc = "upper right", fontsize ="large", ncol=len(task), frameon=False, handlelength = 0.7, columnspacing = 1)
    ax.tick_params(direction="in")
    ax.set_xticks(index_num + 0.25)
    ax.set_ylim([-3500, 3500])
    ax.set_xlabel("Object-COM relation")
    ax.set_ylabel("lag [ms]")
    ax.set_xticklabels(index)

    plt.show()
    fig.savefig(output_dir + "/plot_" + subID + "_lag.png")
    
                    
"""
err = [df.iloc[t*4+4, 0], df.iloc[t*4+4, 1], df.iloc[t*4+4, 2], df.iloc[t*4+4, 3], df.iloc[t*4+4, 4], df.iloc[t*4+4, 5]]
            # taskごとに平均を計算していく        
            for i in range(task_num):
                # 使用するデータのみを抽出
                df_ori = np.array(df[(i*(g.attempt+1))+1 : (i+1)*(g.attempt+1)], dtype = "float")
                mean_all = np.empty([])
 
            # データをDataFrame型に戻す
            result = pd.DataFrame(result_np, columns=["Lcop", "SDx", "SDy", "Srect", "Srms","Ssd"], index=g.task)
            
            # 新しいファイルに結果を書き込む。
            new_file_name = file_name.split('.')[0] + '_means.csv'
            new_file_path = os.path.join(output_dir, new_file_name)
            result.to_csv(new_file_path)

            # 出力フォルダに新しいファイルを保存
            print(f'Saved file: {new_file_path}')
            
            index_num = np.arange(len(result.columns))
            
"""            