# input_folder_path =  "C:/Users/ShimaLab/Desktop/test/Sub5"
# output_folder_path =  "C:/Users/ShimaLab/Desktop/test/Sub5/Sub5"

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 入力フォルダのパス
input_folder_path = "D:/User/kanai/Data/240601/sub1/csv/COP/"

# 出力フォルダの名前
output_folder_name = "COP_Standard"

# 出力フォルダのパス
output_folder_path = os.path.join(input_folder_path, output_folder_name)

# 出力フォルダが存在しない場合、新しく作成する
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 入力フォルダ内のすべてのCSVファイルを処理
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv'):
        # 入力CSVファイルのパス
        input_file_path = os.path.join(input_folder_path, filename)
        
        # CSVファイルをDataFrameとして読み込む
        df = pd.read_csv(input_file_path)
        
        # 列ごとに標準化
        scaler = StandardScaler()
        df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        # 標準化したデータを新しいCSVファイルとして保存
        output_file_path = os.path.join(output_folder_path, filename)
        df_std.to_csv(output_file_path, index=False)

