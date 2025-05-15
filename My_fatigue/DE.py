import numpy as np
from scipy.signal import butter, lfilter
import os
from scipy.io import loadmat
import math


def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# calculate DE
def calculate_DE(saw_EEG_signal):
    variance = np.var(saw_EEG_signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# filter for 5 frequency bands
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(file):
    # read data  sample * channel [1416000, 17]
    frequency = 1000
    data = loadmat(file)["EEG"]["data"][0][0]
    data = data.transpose([1,0])
    data = np.concatenate((data[:100 * frequency, :], data[-100 * frequency:, :]), axis=0)  # (31, 200000)
    print(data.shape)
    # sampling rate

    # samples 1416000
    samples = data.shape[0]
    # 100 samples = 1 DE
    num_sample = int(samples/1000)
    channels = data.shape[1]
    bands = 5
    # init DE [141600, 17, 5]
    DE_3D_feature = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4,   frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8,   frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14,  frequency, order=3)
        beta  = butter_bandpass_filter(trial_signal, 14, 30, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 30, 48, frequency, order=3)
        # DE
        DE_delta = np.zeros(shape=[0], dtype=float)
        DE_theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)
        # DE of delta, theta, alpha, beta and gamma
        for index in range(num_sample):
            DE_delta = np.append(DE_delta, calculate_DE(delta[index * 1000:(index + 1) * 1000]))
            DE_theta = np.append(DE_theta, calculate_DE(theta[index * 1000:(index + 1) * 1000]))
            DE_alpha = np.append(DE_alpha, calculate_DE(alpha[index * 1000:(index + 1) * 1000]))
            DE_beta  = np.append(DE_beta,  calculate_DE(beta[index * 1000:(index + 1) * 1000]))
            DE_gamma = np.append(DE_gamma, calculate_DE(gamma[index * 1000:(index + 1) * 1000]))
        temp_de = np.vstack([temp_de, DE_delta])
        temp_de = np.vstack([temp_de, DE_theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

    temp_trial_de = temp_de.reshape(-1, 5, num_sample)
    temp_trial_de = temp_trial_de.transpose([2, 0, 1])
    DE_3D_feature = np.vstack([temp_trial_de])

    return DE_3D_feature


if __name__ == '__main__':
    # 加载每次选取30s视频后的原始数据文件
    folder_path = 'E:/博士成果/NIPS论文/My_fatigue/sesson3/'
    # 获取指定文件夹下的所有.mat文件名,os.listdir()函数用于获取指定文件夹下的所有文件名
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    i = 1
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        # mat_file = np.load(file_path)
        print('processing {}'.format(file_path))

        # 计算每个被试的皮尔逊相关系数
        Pearson_feature = decompose_to_DE(file_path)
        print(Pearson_feature.shape)
        #单被试结果
        np.save(f"E:/博士成果/NIPS论文/My_fatigue/DE/3/{i}.npy", Pearson_feature)
        i = i + 1