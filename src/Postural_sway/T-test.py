# %%
# # -*- coding: utf-8 -*-
"""
Created on: 2025-01-10 13:02

@author: ShimaLab
"""

import re
import glob
import math
import numpy as np
import pandas as pd
import statistics as st
from scipy import stats
from scipy.stats import rankdata

import global_value as g

"""
変更点
271:検定の種類
300:外れ値の指定
"""

# 実行
def run():
    taskList = g.task
    kindOfTtest = decide_kind_of_Ttest()
    path = "D:/User/kanai/Data/%s/result_COP/result/*.csv" % (g.datafile)
    results = (
        []
    )  # resultまとめファイル(dataframeのlist(3次元)) 例 results[0]=DataFrame, results[1]=DataFrame
    subList = []  # 被験者列(list)
    trialList = []  # 試行回数列(list)
    taskNum = len(taskList)  # タスク数

    # 正規化前後全体の平均, SDのリスト
    wholeAverageBeforeNormalization = []
    wholeAverageAfterNormalization = []
    wholeSDBeforeNormalization = []
    wholeSDAfterNormalization = []

    (subNum, trialNum) = preparation(path, results)# 1次元：被験者番号，2次元：resultの縦，3次元：resultの横
    
    df1 = create_header(taskNum, taskList)
    (subList, trialList) = create_index(subNum, trialNum, subList, trialList)

    # 元データを整理して格納するndarray配列
    totalDxy = np.empty((0, subNum * trialNum), float)  # 総軌跡長
    rmsX = np.empty((0, subNum * trialNum), float)  # SDx
    rmsY = np.empty((0, subNum * trialNum), float)  # SDy
    ampWidthX = np.empty((0, subNum * trialNum), float)  # 振幅幅x
    ampWidthY = np.empty((0, subNum * trialNum), float)  # 振幅幅y
    rectangleArea = np.empty((0, subNum * trialNum), float)  # 矩形面積
    rmsArea = np.empty((0, subNum * trialNum), float)  # 実効値面積
    sdArea = np.empty((0, subNum * trialNum), float)  # 標準偏差面積

    # 正規化後のデータを格納するndarray配列
    normalizeTotalDxy = np.empty((0, subNum * trialNum), float)  # 総軌跡長
    normalizeRmsX = np.empty((0, subNum * trialNum), float)  # SDx
    normalizeRmsY = np.empty((0, subNum * trialNum), float)  # SDy
    normalizeAmpWidthX = np.empty((0, subNum * trialNum), float)  # 振幅幅x
    normalizeAmpWidthY = np.empty((0, subNum * trialNum), float)  # 振幅幅y
    normalizeRectangleArea = np.empty((0, subNum * trialNum), float)  # 矩形面積
    normalizeRmsArea = np.empty((0, subNum * trialNum), float)  # 実効値面積
    normalizeSdArea = np.empty((0, subNum * trialNum), float)  # 標準偏差面積

    # 解析結果のcsvファイルにおける各指標の列(0~18)
    totalDxyColumn = 7  # 総軌跡長
    rmsXColumn = 11  # SDx
    rmsYColumn = 12  # SDy
    ampWidthXColumn = 14  # 振幅幅x
    ampWidthYColumn = 15  # 振幅幅y
    rectangleAreaColumn = 16  # 矩形面積
    rmsAreaColumn = 17  # 実効値面積
    sdAreaColumn = 18  # 標準偏差面積

    totalDxyDf = loop(
        subNum,
        taskNum,
        trialNum,
        totalDxy,
        results,
        totalDxyColumn,
        normalizeTotalDxy,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    rmsXDf = loop(
        subNum,
        taskNum,
        trialNum,
        rmsX,
        results,
        rmsXColumn,
        normalizeRmsX,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    rmsYDf = loop(
        subNum,
        taskNum,
        trialNum,
        rmsY,
        results,
        rmsYColumn,
        normalizeRmsY,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    ampWidthXDf = loop(
        subNum,
        taskNum,
        trialNum,
        ampWidthX,
        results,
        ampWidthXColumn,
        normalizeAmpWidthX,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    ampWidthYDf = loop(
        subNum,
        taskNum,
        trialNum,
        ampWidthY,
        results,
        ampWidthYColumn,
        normalizeAmpWidthY,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    rectangleAreaDf = loop(
        subNum,
        taskNum,
        trialNum,
        rectangleArea,
        results,
        rectangleAreaColumn,
        normalizeRectangleArea,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    rmsAreaDf = loop(
        subNum,
        taskNum,
        trialNum,
        rmsArea,
        results,
        rmsAreaColumn,
        normalizeRmsArea,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    sdAreaDf = loop(
        subNum,
        taskNum,
        trialNum,
        sdArea,
        results,
        sdAreaColumn,
        normalizeSdArea,
        subList,
        trialList,
        df1,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
        kindOfTtest,
    )
    wholeBeforeNormalization = create_whole(
        taskNum, taskList, wholeAverageBeforeNormalization, wholeSDBeforeNormalization
    )
    wholeAfterNormalization = create_whole(
        taskNum, taskList, wholeAverageAfterNormalization, wholeSDAfterNormalization
    )

    outputPath = "D:/User/kanai/Data/%s/result_COP/result/T test_" % (g.datafile)

    if kindOfTtest == 1:
        outputPath += "Paired"
    elif kindOfTtest == 2:
        outputPath += "Welch"
    elif kindOfTtest == 3:
        outputPath += "Brunner-Munzel"

    outputPath += "_sub" + str(subNum) + ".xlsx"
    output_file(
        outputPath,
        wholeBeforeNormalization,
        wholeAfterNormalization,
        totalDxyDf,
        rmsXDf,
        rmsYDf,
        ampWidthXDf,
        ampWidthYDf,
        rectangleAreaDf,
        rmsAreaDf,
        sdAreaDf,
        "正規化前全体",
        "正規化後全体",
        "総軌跡長",
        "SDx",
        "SDy",
        "振幅幅x",
        "振幅幅y",
        "矩形面積",
        "実効値面積",
        "標準偏差面積",
    )

    print("プログラム終了")


""""""

""""""


# t検定の種類を決定
def decide_kind_of_Ttest():
    while True:
#        print("t検定の種類を入力してください")
#        print("1. 対応のあるt検定 , 2. Welchのt検定 , 3. Brunner-Munzel検定")
#        kindOfTtest = int(input())
        kindOfTtest = 1#対応のあるt検定を選択しておく
        if 1 <= kindOfTtest <= 3:
            break

    return kindOfTtest


""""""

# for文では, i→subNum, j→trialNum, k→taskNum としている.

""""""

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# 下処理
def preparation(path, results):
    file = sorted(glob.glob(path), key=natural_keys)  # ディレクトリ下のファイル取得
    subNum = len(file)  # 被験者数
    tasknum = len(g.task)

    # 文字がある行削除
    for i in range(subNum):
        tmpFile = pd.read_csv(file[i], header=None, encoding="shift-jis")
        ##
        #外れ値をここで削除
        ##
        """
        if i==0:
            tmpFile.iloc[28,4:] = np.nan
            tmpFile.iloc[7,4:] = np.nan
        if i==2:
            tmpFile.iloc[19,4:] = np.nan
        if i==3:
            tmpFile.iloc[9,4:] = np.nan
        if i==6:
            tmpFile.iloc[20,4:] = np.nan
        if i==9:
            tmpFile.iloc[20,4:] = np.nan


        """
        """
        # 各タスクの最大最小をはじく
        for k in range(tasknum):
            max_id = tmpFile.iloc[k*(tasknum+1)+1 : (k+1)*(tasknum+1) , 17].idxmax()
            min_id = tmpFile.iloc[k*(tasknum+1)+1 : (k+1)*(tasknum+1) , 17].idxmin()
            
            tmpFile.iloc[max_id,4:] = np.nan
            tmpFile.iloc[min_id,4:] = np.nan
        """
        results.append(tmpFile[~pd.to_numeric(tmpFile[0], errors="coerce").isnull()])

    trialNum = int(max(results[0][1]))  # 試行回数

    return (subNum, trialNum)


""""""

""""""


# 1行目(ヘッダ)作成
def create_header(taskNum, taskList):
    row = ["subNum", "trialNum"]
    # タスク名をリストに格納
    for k in range(taskNum):
        row.append(taskList[k])

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            row += [" "]

    # 正規化前後の間に空列
    empty = [" "]
    header = row + empty + row

    # DataFrameへ
    df = pd.DataFrame(header)
    df = df.T

    return df


""""""

""""""


# subColumn,trialColumnの作成
def create_index(subNum, trialNum, subList, trialList):
    # subNum, trialNumの列作成
    for i in range(subNum):
        for j in range(trialNum):
            subList.append(i + 1)
            trialList.append(j + 1)

    # 二重にする
    subList = [subList]
    trialList = [trialList]

    return (subList, trialList)


""""""

""""""


# 元データの整理
def organize_original(subNum, taskNum, trialNum, value, results, column):
    tmp = []
    for k in range(taskNum):
        for i in range(subNum):
            for j in range(trialNum):
                tmp.append(float(results[i].iat[trialNum * k + j, column]))

        value = np.append(value, [tmp], axis=0)
        tmp.clear()

    return value


""""""

""""""


# NCの平均を求める
def average_NC(subNum, trialNum, results, column):
    # 初期化
    ave = [0] * subNum

    # 平均を求める
    for i in range(subNum):
        trial = trialNum
        for j in range(trialNum):
            num = float(results[i].iat[j, column])
            # nanがあったら無視する機能
            if math.isnan(num):
                trial = trial - 1
            else:
                ave[i] += num
        ave[i] /= trial

    return ave


""""""

""""""


# NCの値で正規化
def normalize(subNum, taskNum, trialNum, value, normalization, ave):
    normalization = value.copy()
    for k in range(taskNum):
        for i in range(subNum * trialNum):
            normalization[k, i] /= ave[i // trialNum]

    return normalization


""""""

""""""


# 被験者ごとの結果
def individual(subNum, taskNum, trialNum, value):
    value = value.tolist()
    zeroList = [0] * subNum
    individual = []
    SD = []
    tmp = zeroList.copy()
    tmpSD = []
    for k in range(taskNum):#0
        for i in range(subNum):#被験者
            attempt = trialNum
            for j in range(trialNum):#0
                num = value[k][j + i * trialNum]
                if math.isnan(num):
                    attempt = attempt-1
                else:
                    tmp[i] += num
            tmp[i] /= attempt
            tmpSD.append(np.nanstd(value[k][i * trialNum : i * trialNum + trialNum]))#, ddof=1
        individual.append(tmp)
        SD.append(tmpSD.copy())
        tmp = zeroList.copy()
        tmpSD.clear()

    return individual, SD


""""""

""""""


# valueのdataframe作成
def create_dataframe(
    value, normalization, subList, trialList, subNum, trialNum, taskNum
):
    # ndarray(numpy)⇔list変換
    value = value.tolist()
    normalization = normalization.tolist()

    # 空白リスト
    emptyList = [[" "] * (subNum * trialNum)]

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            value += emptyList

    # リスト結合
    value = (
        subList + trialList + value + emptyList + subList + trialList + normalization
    )

    # dataframe作成
    df = pd.DataFrame(value)

    # 転置
    df = df.T

    return df


""""""

""""""


# 各値の平均を求める
def average(value, taskNum):
    ave = []
    for k in range(taskNum):
        ave.append(np.nanmean(value[k]))

    return ave


""""""

""""""


# 各値のSDを求める
def standard_deviation(value, taskNum):
    SD = []
    for k in range(taskNum):
        SD.append(np.nanstd(value[k]))#, ddof = 1

    return SD


""""""

""""""


# variousValuesのdataframeの作成
def create_variousValues(
    value,
    normalization,
    taskNum,
    taskList,
    wholeAverageBeforeNormalization,
    wholeAverageAfterNormalization,
    wholeSDBeforeNormalization,
    wholeSDAfterNormalization,
):
    variousValues = []

    # subNum, trialNum列の分を空白にする
    header = [" ", " "]

    # タスク名をリストに追加
    header += taskList

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            header += [" "]

    # 正規化前後の間に空列
    header = header + [" "] + header

    # 平均の行の各値を算出
    averageList = ["平均", " "]
    aveValue = average(value, taskNum)
    aveNormalization = average(normalization, taskNum)

    # 正規化前後全体のリストにも格納
    wholeAverageBeforeNormalization += [aveValue.copy()]
    wholeAverageAfterNormalization += [aveNormalization.copy()]

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            aveValue += [" "]
            aveNormalization += [" "]

    # 平均の行を結合
    averageList = averageList + aveValue + [" "] + averageList + aveNormalization

    # SDの行の各値を算出
    SDList = ["SD", " "]
    SDValue = standard_deviation(value, taskNum)
    SDNormalization = standard_deviation(normalization, taskNum)

    # 正規化前後全体のリストにも格納
    wholeSDBeforeNormalization += [SDValue.copy()]
    wholeSDAfterNormalization += [SDNormalization.copy()]

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            SDValue += [" "]
            SDNormalization += [" "]

    # SDの行を結合
    SDList = SDList + SDValue + [" "] + SDList + SDNormalization

    # ヘッダ, 平均, SD の行を結合
    variousValues.append(header)
    variousValues.append(averageList)
    variousValues.append(SDList)

    # DataFrameへ
    df = pd.DataFrame(variousValues)

    return df


""""""

""""""


# p値を求める
def p_value(value, taskNum, kindOfTtest):
    pValue = []

    # 自分以外のタスクとのp値算出
    for i in range(taskNum):
        for j in range(i + 1, taskNum):
            if kindOfTtest == 1:
                a = pd.DataFrame(value[i]).dropna()
                b = pd.DataFrame(value[j]).dropna()
                # 対応のあるt検定
                T, P = stats.ttest_rel(a, b)
                pValue.append(P)
            elif kindOfTtest == 2:
                # welchのt検定
                T, P = stats.ttest_ind(value[i], value[j], equal_var=False)
                pValue.append(P)
            elif kindOfTtest == 3:
                # Brunner-Munzelのt検定
                T, P = stats.brunnermunzel(value[i], value[j])
                pValue.append(P)

    return pValue


""""""

""""""


# p値の順位を求める
def p_rank(value, taskNum, kindOfTtest):
    # 対象のデータ
    pValue = p_value(value, taskNum, kindOfTtest)

    # 降順
    pRank = rankdata(-np.array(pValue)).astype(int)

    return pRank


""""""

""""""


# holm法
def holm_method(value, taskNum, kindOfTtest):
    holm = []

    # p値算出
    pValue = p_value(value, taskNum, kindOfTtest)

    # p値の順位(降順)算出
    pRank = p_rank(value, taskNum, kindOfTtest)

    # holm法(p値×順位)
    for i in range(len(pValue)):
        holm.append(pValue[i] * pRank[i])

    return holm


""""""

""""""


# pValueのlist作成
def create_pList(value, normalization, taskNum, taskList, kindOfTtest):
    # 変数
    task1 = []
    task2 = []
    emptyList = []

    # タスク1,2の列と空列の作成
    for i in range(taskNum):
        for j in range(i + 1, taskNum):
            task1.append(taskList[i])
            task2.append(taskList[j])
            emptyList.append(" ")

    # 各値の算出

    # 正規化前
    pValue = p_value(value, taskNum, kindOfTtest)
    pRank = p_rank(value, taskNum, kindOfTtest).tolist()
    holm = holm_method(value, taskNum, kindOfTtest)

    # 正規化後
    pValue_normalization = p_value(normalization, taskNum, kindOfTtest)
    pRank_normalization = p_rank(normalization, taskNum, kindOfTtest).tolist()
    holm_normalization = holm_method(normalization, taskNum, kindOfTtest)

    # holmのみ二重に
    holm = [holm, [i * 100 for i in holm]]
    holm_normalization = [holm_normalization, [i * 100 for i in holm_normalization]]

    # タスク数に応じて空列調整
    if taskNum > 4:
        for k in range(taskNum - 4):
            holm += [emptyList]

    # 結合
    pList = (
        [task1]
        + [task2]
        + [pValue]
        + [pRank]
        + holm
        + [emptyList]
        + [task1]
        + [task2]
        + [pValue_normalization]
        + [pRank_normalization]
        + holm_normalization
    )

    return pList


""""""

""""""


# pFrameのdataframeの作成
def create_pFrame(value, normalization, taskNum, taskList, kindOfTtest):
    # ヘッダの初期化
    header = ["タスク1", "タスク2", "p値", "順位", "ホルム法", "ホルム法[%]"]

    # タスク数に応じて空列調整
    if taskNum > 4:
        for k in range(taskNum - 4):
            header += [" "]

    # 正規化前後の間に空列
    header = header + [" "] + header

    # 二重に
    header = [header]

    # p値, 順位, holm法の算出
    pList = create_pList(value, normalization, taskNum, taskList, kindOfTtest)

    # ヘッダのDataFrame
    df1 = pd.DataFrame(header)

    # 各値のDataFrame
    df2 = pd.DataFrame(pList)

    # 転置
    df2 = df2.T

    # DataFrame の結合
    df = pd.concat([df1, df2], axis=0)

    return df


""""""

""""""


# individualのdataframeの作成
def create_individualFrame(value, normalization, subNum, taskNum, trialNum, taskList):
    individualFrame = []
    emptyList = [" "] * subNum
    subList = ["sub"] * subNum

    # subNum, trialNum列の分を空白にする
    resultHeader = ["結果", " "]
    SDHeader = ["SD", " "]

    # タスク名をリストに追加
    resultHeader += taskList
    SDHeader += taskList

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            resultHeader += [" "]
            SDHeader += [" "]

    # 正規化前後の間に空列
    resultHeader = resultHeader + [" "] + resultHeader
    SDHeader = SDHeader + [" "] + SDHeader

    # subList作成
    for i in range(subNum):
        subList[i] = subList[i] + str(i + 1)

    # 各被験者ごとの結果を求める
    resultValue, SDValue = individual(subNum, taskNum, trialNum, value)
    resultNormalization, SDNormalization = individual(
        subNum, taskNum, trialNum, normalization
    )

    aveValue = average(value, taskNum)
    aveNormalization = average(normalization, taskNum)

    # タスク数に応じて空列調整
    if taskNum < 4:
        for k in range(4 - taskNum):
            resultValue += [emptyList]
            SDValue += [emptyList]
            resultNormalization += [emptyList]
            SDNormalization += [emptyList]

    # 結果の行を結合
    result = (
        [emptyList]
        + [subList]
        + resultValue
        + [emptyList]
        + [emptyList]
        + [subList]
        + resultNormalization
    )
    SD = (
        [emptyList]
        + [subList]
        + SDValue
        + [emptyList]
        + [emptyList]
        + [subList]
        + SDNormalization
    )

    # ヘッダ, result, SDをDataFrameへ
    resultHeaderFrame = pd.DataFrame(resultHeader)
    SDHeaderFrame = pd.DataFrame(SDHeader)
    resultFrame = pd.DataFrame(result)
    SDFrame = pd.DataFrame(SD)

    # ヘッダ, result, SDを転置
    resultHeaderFrame = resultHeaderFrame.T
    SDHeaderFrame = SDHeaderFrame.T
    resultFrame = resultFrame.T
    SDFrame = SDFrame.T

    # result, SDともにヘッダと結合
    resultFrame = pd.concat([resultHeaderFrame, resultFrame], axis=0)
    SDFrame = pd.concat([SDHeaderFrame, SDFrame], axis=0)

    # result, SDを結合
    resultFrame.loc["empty"] = " "
    df = pd.concat([resultFrame, SDFrame], axis=0)

    return df


""""""

""""""


# dataframe結合
def combine_dataframe(df1, df2, df3, df4, df5):
    df = pd.concat([df1, df2], axis=0)
    df.loc["empty1"] = " "
    df = pd.concat([df, df3], axis=0)
    df.loc["empty2"] = " "
    df = pd.concat([df, df4], axis=0)
    df.loc["empty3"] = " "
    df = pd.concat([df, df5], axis=0)

    return df


""""""

""""""


# ファイル出力
def output_file(
    path,
    df1,
    df2,
    df3,
    df4,
    df5,
    df6,
    df7,
    df8,
    df9,
    df10,
    sheetName1,
    sheetName2,
    sheetName3,
    sheetName4,
    sheetName5,
    sheetName6,
    sheetName7,
    sheetName8,
    sheetName9,
    sheetName10,
):
    with pd.ExcelWriter(path) as writer:
        df1.to_excel(writer, sheet_name=sheetName1, index=False, header=False)
        df2.to_excel(writer, sheet_name=sheetName2, index=False, header=False)
        df3.to_excel(writer, sheet_name=sheetName3, index=False, header=False)
        df4.to_excel(writer, sheet_name=sheetName4, index=False, header=False)
        df5.to_excel(writer, sheet_name=sheetName5, index=False, header=False)
        df6.to_excel(writer, sheet_name=sheetName6, index=False, header=False)
        df7.to_excel(writer, sheet_name=sheetName7, index=False, header=False)
        df8.to_excel(writer, sheet_name=sheetName8, index=False, header=False)
        df9.to_excel(writer, sheet_name=sheetName9, index=False, header=False)
        df10.to_excel(writer, sheet_name=sheetName10, index=False, header=False)


""""""

""""""


# 繰り返し部分
def loop(
    subNum,
    taskNum,
    trialNum,
    value,
    results,
    column,
    normalization,
    subList,
    trialList,
    df1,
    taskList,
    wholeAverageBeforeNormalization,
    wholeAverageAfterNormalization,
    wholeSDBeforeNormalization,
    wholeSDAfterNormalization,
    kindOfTtest,
):
    # 下処理
    value = organize_original(subNum, taskNum, trialNum, value, results, column)
    # NC の平均算出
    aveNC = average_NC(subNum, trialNum, results, column)

    # NC の値で正規化
    normalization = normalize(subNum, taskNum, trialNum, value, normalization, aveNC)

    # individual(subNum, taskNum, trialNum, value)
    # create_individualFrame(value, normalization, subNum, taskNum, trialNum,taskList)
    # valueのdataframe作成
    df2 = create_dataframe(
        value, normalization, subList, trialList, subNum, trialNum, taskNum
    )

    # variousValuesのdataframeの作成
    df3 = create_variousValues(
        value,
        normalization,
        taskNum,
        taskList,
        wholeAverageBeforeNormalization,
        wholeAverageAfterNormalization,
        wholeSDBeforeNormalization,
        wholeSDAfterNormalization,
    )

    # pFrameのdataframeの作成
    df4 = create_pFrame(value, normalization, taskNum, taskList, kindOfTtest)

    # individualのdataframeの作成
    df5 = create_individualFrame(
        value, normalization, subNum, taskNum, trialNum, taskList
    )

    # dataframe結合
    df = combine_dataframe(df1, df2, df3, df4, df5)

    return df


""""""

""""""


# 正規化前, 後の全体の作成
def create_whole(taskNum, taskList, wholeAverage, wholeSD):
    # 1行目作成
    averageFirstLine = ["平均"]
    SDFirstLine = ["SD"]
    empty = [" "]

    # タスク数の分だけ, 空白
    for i in range(taskNum):
        averageFirstLine += empty
        SDFirstLine += empty

    # 1行目結合
    firstLine = averageFirstLine + empty + SDFirstLine

    # 2行目作成
    secondLine = ["指標"]

    # タスク名をリストに追加
    secondLine += taskList

    # 2行目結合
    secondLine += empty + secondLine

    # 各指標作成
    # 平均
    totalDxyAverage = ["L_COP"]
    rmsXAverage = ["SDx"]
    rmsYAverage = ["SDy"]
    ampWidthXAverage = ["x_amp"]
    ampWidthYAverage = ["y_amp"]
    rectangleAreaAverage = ["S_rect"]
    rmsAreaAverage = ["S_rms"]
    sdAreaAverage = ["S_SD"]

    # 値をリストに追加
    totalDxyAverage += wholeAverage[0]
    rmsXAverage += wholeAverage[1]
    rmsYAverage += wholeAverage[2]
    ampWidthXAverage += wholeAverage[3]
    ampWidthYAverage += wholeAverage[4]
    rectangleAreaAverage += wholeAverage[5]
    rmsAreaAverage += wholeAverage[6]
    sdAreaAverage += wholeAverage[7]

    # SD
    totalDxySD = ["L_COP"]
    rmsXSD = ["SDx"]
    rmsYSD = ["SDy"]
    ampWidthXSD = ["x_amp"]
    ampWidthYSD = ["y_amp"]
    rectangleAreaSD = ["S_rect"]
    rmsAreaSD = ["S_rms"]
    sdAreaSD = ["S_SD"]

    # 値をリストに追加
    totalDxySD += wholeSD[0]
    rmsXSD += wholeSD[1]
    rmsYSD += wholeSD[2]
    ampWidthXSD += wholeSD[3]
    ampWidthYSD += wholeSD[4]
    rectangleAreaSD += wholeSD[5]
    rmsAreaSD += wholeSD[6]
    sdAreaSD += wholeSD[7]

    # 平均とSD結合
    totalDxy = totalDxyAverage + empty + totalDxySD
    rmsX = rmsXAverage + empty + rmsXSD
    rmsY = rmsYAverage + empty + rmsYSD
    ampWidthX = ampWidthXAverage + empty + ampWidthXSD
    ampWidthY = ampWidthYAverage + empty + ampWidthYSD
    rectangleArea = rectangleAreaAverage + empty + rectangleAreaSD
    rmsArea = rmsAreaAverage + empty + rmsAreaSD
    sdArea = sdAreaAverage + empty + sdAreaSD

    # 全結合
    whole = (
        [firstLine]
        + [secondLine]
        + [totalDxy]
        + [rmsX]
        + [rmsY]
        + [ampWidthX]
        + [ampWidthY]
        + [rectangleArea]
        + [rmsArea]
        + [sdArea]
    )

    # DataFrameへ
    df = pd.DataFrame(whole)

    return df


""""""

""""""
run()

""""""
