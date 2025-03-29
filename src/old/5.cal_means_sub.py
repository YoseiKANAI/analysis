# %%
# -*- coding: utf-8 -*-
# 被験者1名に対してのCOPを算出
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

input_dir = "D:/User/kanai/Data/%s/result_COP/result" %(g.datafile)
output_dir = "D:/User/kanai/Data/%s/result_COP/COP_index" %(g.datafile)
task_num = len(g.task)
# 出力先フォルダを作成
os.makedirs(output_dir, exist_ok=True)

# sammary用の値
index = "Ssd"
result_summary = np.empty((g.subnum, task_num))
std_summary = np.empty((g.subnum, task_num))
ID = 0

for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith('.csv'):
            # CSVファイルを読み込む
            file_path = os.path.join(root, file_name)
            df = pd.read_csv(file_path, encoding="cp932", header = None)
            
            column_means = np.empty([0])
            
            std_L = []
            std_SDx = []
            std_SDy = []
            std_Srect = []
            std_Srms = []
            std_Ssd = []
            
            # taskごとに平均を計算していく        
            for i in range(task_num):
                # 使用するデータのみを抽出
                df_ori = np.array(df[(i*(g.attempt+1))+1 : (i+1)*(g.attempt+1)], dtype = "float")
                mean_all = np.empty([])
                
                # 目標指標値を抽出し，array型に変換
                # L：総軌跡長，SD：標準偏差x,y，S：各面積指標
                df_L = df_ori[:,7]
                df_SDx = df_ori[:, 11]
                df_SDy = df_ori[:, 12]
                df_Srect = df_ori[: ,16]
                df_Srms = df_ori[: ,17]
                df_Ssd = df_ori[: ,18]
                
                
                # 各指標の分散を算出
                std_L.append(np.std(df_L))
                std_SDx.append(np.std(df_SDx))
                std_SDy.append(np.std(df_SDy))
                std_Srect.append(np.std(df_Srect))
                std_Srms.append(np.std(df_Srms))
                std_Ssd.append(np.std(df_Ssd))
                
                # 各指標の平均を算出
                mean_L = np.mean(df_L)
                mean_SDx = np.mean(df_SDx)
                mean_SDy = np.mean(df_SDy)
                mean_Srect = np.mean(df_Srect)
                mean_Srms = np.mean(df_Srms)
                mean_Ssd = np.mean(df_Ssd)
                
                # 各指標を一つにまとめる
                mean_all = np.append(mean_L, mean_SDx)
                mean_all = np.append(mean_all, mean_SDy)
                mean_all = np.append(mean_all, mean_Srect)
                mean_all = np.append(mean_all, mean_Srms)
                mean_all = np.append(mean_all, mean_Ssd)
                
                # 1task分のデータをcolumn_meansに格納
                column_means = np.append(column_means, mean_all)                

            # 得られたデータを[4,6]のデータに変形し格納
            result_np = column_means.reshape([task_num, 6])
            
            # NC条件で割り，標準偏差を正規化に合わせる                 
            std_L = std_L/result_np[:,0]
            std_SDx = std_SDx/result_np[:,1]
            std_SDy = std_SDy/result_np[:,2]
            std_Srect = std_Srect/result_np[:,3]
            std_Srms = std_Srms/result_np[:,4]
            std_Ssd = std_Ssd/result_np[:,5]
            
            # NC条件で正規化する
            result_np = result_np / result_np[0, :]
            
            # データを連結用の配列に格納　-> Ssd
            result_summary[ID, :] = result_np[:, 5]
            std_summary[ID, :] = std_Ssd
            
            # データをDataFrame型に戻す
            result = pd.DataFrame(result_np, columns=["Lcop", "SDx", "SDy", "Srect", "Srms","Ssd"], index=g.task)
            
            # 新しいファイルに結果を書き込む。
            new_file_name = file_name.split('.')[0] + '_means.csv'
            new_file_path = os.path.join(output_dir, new_file_name)
            result.to_csv(new_file_path)

            # 出力フォルダに新しいファイルを保存
            print(f'Saved file: {new_file_path}')
            
            index_num = np.arange(len(result.columns))
            
            # グラフをプロット
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams["font.size"] = 12   
            
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            for i in range(task_num):
                slide = i*0.17
                err = [std_L[i], std_SDx[i], std_SDy[i], std_Srect[i], std_Srms[i], std_Ssd[i]]
                ax.bar(index_num+slide, result.iloc[i,:], width=0.17, yerr=err, capsize=3, label = g.task[i])
            ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
            ax.tick_params(direction="in")
            ax.set_xticks(index_num + 0.3)
            ax.set_ylim([0.0, 2.0])
            ax.set_ylabel("Average values normalized \n By NC values")
            ax.set_xticklabels(["$L_{COP}$", "$\sigma_{x}$", "$\sigma_{y}$", "$S_{rect}$", "$S_{rms}$", "$S_{\sigma}$"])
            
            #plt.show()
            fig.savefig(output_dir + "/plot_" + file_name.replace("result.csv","") + ".png")
            ID = ID+1

# 全被験者のグラフを横並びにプロット

sublist = []
for i in range(g.subnum):
    sub = "Sub %d" %(i+1)
    sublist.append(sub)
    
tasklist = ["NC","FB","DBmass","DBchar", "DW"]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sub_num = np.arange(g.subnum)
for i in range(task_num):
    slide = i*0.15
    err = [std_summary[:, i]]
    ax.bar(sub_num+slide, result_summary[:,i], width=0.12
           , yerr=err, capsize=3, label = tasklist[i])
ax.legend(loc = "upper right", fontsize ="large", ncol=task_num, frameon=False, handlelength = 0.7, columnspacing = 1)
ax.tick_params(direction="in")
ax.set_xticks(sub_num + 0.3)
ax.set_ylim([0.0, 2.0])
ax.set_ylabel("Average values normalized \n By NC values")
ax.set_xticklabels(sublist)
#plt.show()
output_filename = "/plot_"+ index +"_summary.png"
fig.savefig(output_dir + output_filename)