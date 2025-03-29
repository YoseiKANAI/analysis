# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal, fft


class SignalProcessor:
    # 筋電データのファイル名
    raw_signals_filename = 'raw_signals.txt'

    # 各チャネルの筋電のオフセット（フィルタ処理済み）
    offset_EMG = np.array([0.00378768,
                           0.00324894,
                           0.00419124,
                           0.00405966])
    
    # 各チャネルの最大筋電量（フィルタ処理済み）
    max_EMG = np.array([0.0860999,
                        0.260329,
                        0.121923,
                        0.188236])
    
    # サンプリング周波数
    sampling_freq = 1000
    # ナイキスト周波数
    nyq_freq = sampling_freq / 2
    # バタワースフィルタの次数
    butter_order = 2
    # バンドパスフィルタの通過域端周波数（1~250Hz）
    bp_passband = np.array([1.0, 250])
    # ローパスフィルタのカットオフ周波数（1Hz）
    lp_cutoff = 1.0

    def __init__(self, title='raw_signals'):
        """SingalProcessorクラスのコンストラクタ

        生筋電データを配列raw_signalsに格納し、波形を描画する
        """
        # 筋電データの読み込み
        self.raw_signals = np.loadtxt(SignalProcessor.raw_signals_filename, delimiter='\t', skiprows=1)
        # 筋電データのサンプル数
        self.sample_num = len(self.raw_signals)
        # 筋電データのチャネル数
        self.channel_num = len(self.raw_signals[0])
        # 筋電データの時間軸
        self.time = np.arange(0, self.sample_num / SignalProcessor.sampling_freq, 1 / SignalProcessor.sampling_freq)
        # 波形の描画
        self.plot_signals(self.raw_signals, title)

        return
    
    def bandpass_filter(self, title='bp_filtered_signals'):
        """raw_signalsにバンドパスフィルタをかける

        scipyのバタワースフィルタを使用し、各チャネルの生筋電データにフィルタをかける
        処理後のデータをbp_filtered_signalsに格納し、波形を描画する
        """
        self.bp_filtered_signals = np.empty(self.raw_signals.shape)
        ######################################################################################
        # 以下を実装
        y = np.empty(self.raw_signals.shape)
        b, a = signal.butter(self.butter_order, 0.5, btype='lowpass')
        d, c = signal.butter(self.butter_order, 0.002, btype='highpass')
        
        # フィルタの適用
        for i in range(self.channel_num):
            y[:,i] = signal.filtfilt(b, a, self.raw_signals[:,i])
            self.bp_filtered_signals[:,i] = signal.filtfilt(d, c, y[:,i])
        
        # プロット
        self.plot_signals(self.bp_filtered_signals, title)
        ######################################################################################
        return
    
    
    def rectifier(self, title='rectified_signals'):
        """bp_filtered_signalsに対して全波整流（絶対値算出）を行う

        処理後のデータをrectified_signalsに格納し、波形を描画する
        """
        self.rectified_signals = np.empty(self.bp_filtered_signals.shape)
        ######################################################################################
        # 以下を実装
        self.rectified_signals = np.abs(self.bp_filtered_signals)
        self.plot_signals(self.rectified_signals, title)
        ######################################################################################

        return
    
    
    def lowpass_filter(self, title='lp_filtered_signals'):
        """rectified_signalsにローパスフィルタをかける

        scipyのバタワースフィルタを使用し、各チャネルのデータにフィルタをかける
        処理後のデータをlp_filtered_signalsに格納し、波形を描画する
        """
        self.lp_filtered_signals = np.empty(self.rectified_signals.shape)
        ######################################################################################
        # 以下を実装
        b, a = signal.butter(self.butter_order, 0.002, btype='lowpass')
        
        # フィルタの適用
        for i in range(self.channel_num):
            self.lp_filtered_signals[:,i] = signal.filtfilt(b, a, self.rectified_signals[:,i])

        self.plot_signals(self.lp_filtered_signals, title)
        ######################################################################################
        return
    
    
    def eliminate_offset(self, title='offset_eliminated_signals'):
        """lp_filtered_signalsからオフセットを除去する

        各チャネルのデータからオフセットの値（offset_EMG）を引く
        計算の結果が0未満となる場合は値を0にする
        処理後のデータをoffset_eliminated_signalsに格納し、波形を描画する
        """
        self.offset_eliminated_signals = np.empty(self.lp_filtered_signals.shape)
        ######################################################################################
        # 以下を実装
        for i in range(self.sample_num):
            self.offset_eliminated_signals[i, :] = self.lp_filtered_signals[i, :] - self.offset_EMG
            for k in range(self.channel_num):
                if self.offset_eliminated_signals[i, k] < 0:
                    self.offset_eliminated_signals[i, k] = 0
        
        self.plot_signals(self.offset_eliminated_signals, title)
        ######################################################################################
        return


    def normalize_by_max(self, title='max_normalized_signals'):
        """offset_eliminated_signalsを最大筋電量で正規化し、発揮量を算出する

        最大筋電量（max_EMG）が1となるように、各チャネルのデータをmax_EMGとoffset_EMGの差で割る
        計算の結果が1を超える場合は値を1にする
        処理後のデータをmax_normalized_signalsに格納し、波形を描画する
        """
        self.max_normalized_signals = np.empty(self.offset_eliminated_signals.shape)
        ######################################################################################
        # 以下を実装
        diff = self.max_EMG - self.offset_EMG
        
        for i in range(self.sample_num):
            self.max_normalized_signals[i, :] = self.offset_eliminated_signals[i, :] / diff
            
        self.plot_signals(self.max_normalized_signals, title)
        ######################################################################################
        return


    def set_power_threshold(self, title='signals_above_threshold'):
        """max_normalized_signalsの総和（4チャネル分）に対してしきい値0.5（50%）を設ける

        全チャネルの発揮量の総和が0.5未満となる部分は安静状態とみなし、全チャネルの値を0にする
        処理後のデータをsignals_above_thresholdに格納し、波形を描画する
        """ 
        self.signals_above_threshold = np.empty(self.max_normalized_signals.shape)
        ######################################################################################
        # 以下を実装
        for i in range(self.sample_num):
            sum = np.sum(self.max_normalized_signals[i, :])
            for k in range(self.channel_num):
                if sum < 0.5:
                    self.signals_above_threshold[i, k] = 0
                else:
                    self.signals_above_threshold[i, k] = self.max_normalized_signals[i, k]

        self.plot_signals(self.signals_above_threshold, title)
        ######################################################################################
        return
    
    
    def normalize_by_sum(self, title='sum_normalized_signals'):
        """signals_above_thresholdの総和（4チャネル分）で正規化し、パターン情報を取得する

        全チャネルチャネルの総和が1となるように、各チャネルのデータを総和で割る
        処理後のデータをsum_normalized_signalsに格納し、波形を描画する
        """
        self.sum_normalized_signals = np.empty(self.signals_above_threshold.shape)
        ######################################################################################
        # 以下を実装
        for i in range(self.sample_num):
            sum = np.sum(self.signals_above_threshold[i, :])
            for k in range(self.channel_num):
                if sum == 0:
                    self.sum_normalized_signals[i, k] = 0
                else:
                    self.sum_normalized_signals[i, k] = self.signals_above_threshold[i, k] / sum

        self.plot_signals(self.sum_normalized_signals, title)
        ######################################################################################
        return
    
    def plot_signals(self, signals, title):
        """波形の描画を行う

        signals:描画する波形の配列
        title:グラフのタイトル
        """
        rows = int((self.channel_num + 3) / 2)
        columns = 2
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig = plt.figure(title, figsize=(columns * 6, rows * 3), tight_layout=True)
        fig.suptitle(title)
        ax_all = fig.add_subplot(rows, 1, rows, title='All Channels', xlabel='Time', ylabel='Amplitude')

        for i in range(self.channel_num):
            ax = fig.add_subplot(rows, columns, i + 1, title=f'Ch-{i + 1}', xlabel='Time', ylabel='Amplitude')
            ax.plot(self.time, signals[:, i], color=color[i])
            ax_all.plot(self.time, signals[:, i], label=f'Ch-{i + 1}')

        ax_all.legend(loc='upper right')

        plt.show()

        return
    
    
    def save_signals(self):
        """信号をファイルに保存する

        signals:保存する信号の配列
        filename:保存するファイル名
        """
        signal_1 = self.sum_normalized_signals[3000:3500,:]
        signal_2 = self.sum_normalized_signals[8000:8500,:]
        signal_3 = self.sum_normalized_signals[17000:17500,:]
        signal_4 = self.sum_normalized_signals[22000:22500,:]
    
        df_1 = pd.DataFrame(signal_1)
        df_2 = pd.DataFrame(signal_2)
        df_3 = pd.DataFrame(signal_3)
        df_4 = pd.DataFrame(signal_4)
        
        df_1.to_csv('./EMG/data_1.csv')
        df_2.to_csv('./EMG/data_2.csv')
        df_3.to_csv('./EMG/data_3.csv')
        df_4.to_csv('./EMG/data_4.csv')
        #np.savetxt(filename, signals, delimiter='\t', header='Ch1\tCh2\tCh3\tCh4', comments='')
        return
    
def main():
    signal_processor = SignalProcessor()
    signal_processor.bandpass_filter()
    signal_processor.rectifier()
    signal_processor.lowpass_filter()
    signal_processor.eliminate_offset()
    signal_processor.normalize_by_max()
    signal_processor.set_power_threshold()
    signal_processor.normalize_by_sum()
    signal_processor.save_signals()

if __name__ == "__main__":
    main()