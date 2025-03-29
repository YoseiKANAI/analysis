# %%
# # -*- coding: utf-8 -*-
# 筋電信号のウェーブレット変換を行うプログラム

"""
Created on: 2024-11-26 16:10

@author: ShimaLab
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statistics as st
import pywt

import global_value as g

Fs = 2000
task_num = len(g.task)

leg_muscle_right = ["SO_R", "TA_R"]
leg_muscle_left = ["SO_L", "TA_L"]
arm_muscle_right = ["MF_L", "EO_L"]
arm_muscle_left = ["MF_R", "EO_R"]

def main():
    for i in range(1):
        ID = 3
        # パスを指定
        input_dir = "D:/User/kanai/Data/%s/sub%s/csv/RawMVC/*.csv" % (g.datafile, ID+1)
        output_dir_indi = output_preparation_indi(ID)
        # 各試行全体のCWTを計算
        CWT(input_dir, ID, output_dir_indi)
        #sub = "sub%d" %(ID+1)
        #CWT_summary = pd.concat([CWT_summary, pd.DataFrame([sub])])
        #CWT_summary = pd.concat([CWT_summary, df])
    #sheet_name = "CWT_whole"
    #excel_output(CWT_summary, output_name, sheet_name)
        
"""
CWTのメインの関数
"""
def CWT(input_dir, ID, output_dir_indi):
    task_list = []
    filelist = glob.glob(input_dir)
    # リストの順に呼び出し
    for t in g.task:
        task_list = [s for s in filelist if t in s]
        #t_data = np.empty((g.attempt, num_windows, g.muscle_num))

        for f in task_list:
            attempt = f[(f.find("\\")+1):(f.find("\\")+7)]
            
            output_dir_indi_plt = output_dir_indi + attempt
            os.makedirs(output_dir_indi_plt, exist_ok=True)
            
            df = pd.read_csv(f)
            """
            if (attempt =="FB0001" or attempt =="DW0004") and ID ==0:
                #CWT = pd.concat([CWT, pd.DataFrame(index=[task_name])])
                continue
            if attempt =="FB0003" and ID ==3:
                #CWT = pd.concat([CWT, pd.DataFrame(index=[task_name])])
                continue
            
            # データがうまく取れてないやつ
            
            if attempt =="NC0002" and ID ==2:
                CWT = pd.concat([CWT, pd.DataFrame(index=[task_name])])
                continue            
            """
            muscle_list = []
            """
            if g.domi_leg[ID] == 0:
                muscle_list = muscle_list + leg_muscle_right
            elif g.domi_leg[ID] == 1:
                muscle_list = muscle_list + leg_muscle_left

            if g.domi_arm[ID] == 0:
                muscle_list = muscle_list + arm_muscle_right
            elif g.domi_arm[ID] == 1:
                muscle_list = muscle_list + arm_muscle_left
            """
            muscle_list = leg_muscle_right+leg_muscle_left+arm_muscle_right+arm_muscle_left
            
            CWT, freqs = CWT_cal(df.interpolate("index"), muscle_list)

            # 各チャンネルのスケログラムを可視化
            for ch in range(len(CWT)):
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["mathtext.fontset"] = "cm"
                plt.rcParams["font.size"] = 12   
                plt.figure(figsize=(12, 6))
                
                if "MF" in muscle_list[ch]:
                    vmax=0.1
                elif "EO" in muscle_list[ch]:
                    vmax=0.2
                elif "SO" in muscle_list[ch]:
                    vmax=0.2
                elif "TA" in muscle_list[ch]:
                    vmax=0.2
                
                plt.imshow(
                    CWT[ch], 
                    extent=[0, 30, 10, 200], 
                    cmap='jet', 
                    aspect='auto', 
                    origin='lower',
                    vmin = 0,
                    vmax = vmax
                )
                
                plt.colorbar(label='Amplitude')
                plt.xlabel('Time (s)')
                plt.ylabel('Scale')
                plt.title(f'Channel {muscle_list[ch]} - Scalogram (CWT Result)')
                plot_name = output_dir_indi_plt + "/plot_" + muscle_list[ch] + ".png"
                plt.savefig(plot_name)
                plt.clf()
                plt.close


"""
CWTを算出
"""
def CWT_cal(df, muscle_list):
    # 中心周波数を設定（例: Morletウェーブレット）
    wavelet = 'cmor' # or "haar"
    central_frequency = 5  # Morletウェーブレットの中心周波数(Haarなら1)

    # スケール範囲を10Hz～500Hzに対応させる
    frequencies = np.arange(10, 201)  # 1Hzから50Hz
    scales = Fs / (frequencies * central_frequency)
  
    cwt_results = []
    
    for ch in muscle_list:
        cwt_result, freqs= pywt.cwt(df[ch], scales, wavelet, sampling_period=1/Fs)
        """
        # エネルギー密度の計算 (絶対値の二乗)
        energy_density = np.abs(cwt_result) ** 2
        # 全エネルギーの計算
        total_energy = np.sum(energy_density)
        # エネルギー密度の正規化
        normalized_energy_density = energy_density / total_energy
        """
        cwt_results.append(np.abs(cwt_result))
    cwt_results = np.array(cwt_results)

    return cwt_results, freqs

"""
CWTを計算, シートに書き込み

def CWT_cal(psd, freq, columns_list):
    CWT = pd.DataFrame([np.zeros(g.muscle_num)], columns = columns_list)
    sum_numr = pd.DataFrame([np.zeros(g.muscle_num)], columns = columns_list)
    sum_deno = pd.DataFrame([np.zeros(g.muscle_num)], columns = columns_list)
    
    for i in range(len(psd)):
        sum_numr += psd[i, :]*freq[i]
        sum_deno += psd[i, :]
        
    CWT = sum_numr / sum_deno
    return CWT
"""

"""
excelに出力
"""
def excel_output(data, output_name, sheet_name):
    if (os.path.isfile(output_name)):
        with pd.ExcelWriter(output_name, mode="a") as writer:
            data.to_excel(writer, sheet_name = sheet_name)
    else:
        with pd.ExcelWriter(output_name) as writer:
            data.to_excel(writer, sheet_name = sheet_name)        

"""
openpyxlを用いてexcel出力
"""
def output_excel_wb(wb, mean, std, output_name, sheet_name):
    # ワークブックとシートの作成
    ws = wb.create_sheet(title=sheet_name)

    for t in range(task_num):
        # list1を書き込む
        for row_idx, row in enumerate(mean[t], start=(len(mean[0])+1)*t+1):
            for col_idx, value in enumerate(row, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)

        # list2を書き込む (list1の右に空白を挟む)
        start_col = len(mean[0][0]) + 2  # list1の列数 + 空白の幅
        for row_idx, row in enumerate(std[t], start=(len(mean[0])+1)*t+1):
            for col_idx, value in enumerate(row, start=start_col):
                ws.cell(row=row_idx, column=col_idx, value=value)
    # Excelファイルとして保存
    wb.save(output_name)


"""
ファイル名の定義
"""        
def output_preparation():
    # ルートフォルダのパスを指定
    output_dir = "D:/User/kanai/Data/%s/result_EMG/CWT" %(g.datafile)
    output_dir_plot = "D:/User/kanai/Data/%s/result_EMG/CWT/plot" %(g.datafile)
    output_name = output_dir + "/result.xlsx"
    # 出力先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)
    # エクセルファイルの初期化
    if (os.path.isfile(output_name)):
        os.remove(output_name)
    
    return output_name, output_dir_plot

"""
個別ファイルの格納先指定
"""  
def output_preparation_indi(ID):
    output_dir = "D:/User/kanai/Data/%s/sub%d/csv/CWT/" %(g.datafile, ID+1)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

if __name__ == "__main__":
    main()
# %%
