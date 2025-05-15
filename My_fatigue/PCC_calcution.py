import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from scipy import stats
import os


#PCC计算

def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# filter for 5 frequency bands
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y



def decompose_to_correlation(file):
    # read data  sample * channel [1416000, 17]
    data = loadmat(file)["EEG"]["data"][0][0]
    # sampling rate
    frequency = 1000
    # samples 1416000

    channels = data.shape[1]

    filter_delta = []
    filter_theta = []
    filter_alpha = []
    filter_beta = []
    filter_gamma = []

    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4,   frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8,   frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14,  frequency, order=3)
        beta  = butter_bandpass_filter(trial_signal, 14, 30, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 30, 48, frequency, order=3)

        filter_delta.append(delta)
        filter_delta = np.array(filter_delta)
        filter_delta = filter_delta.tolist()

        filter_theta.append(theta)
        filter_theta = np.array(filter_theta)
        filter_theta = filter_theta.tolist()

        filter_alpha.append(alpha)
        filter_alpha = np.array(filter_alpha)
        filter_alpha = filter_alpha.tolist()

        filter_beta.append(beta)
        filter_beta = np.array(filter_beta)
        filter_beta = filter_beta.tolist()

        filter_gamma.append(gamma)
        filter_gamma = np.array(filter_gamma)
        filter_gamma = filter_gamma.tolist()

    #(62,N)
    filter_delta = np.array(filter_delta)
    filter_theta = np.array(filter_theta)
    filter_alpha = np.array(filter_alpha)
    filter_beta = np.array(filter_beta)
    filter_gamma = np.array(filter_gamma)

    #(N,62)
    filter_delta = filter_delta.T
    filter_theta = filter_theta.T
    filter_alpha = filter_alpha.T
    filter_beta = filter_beta.T
    filter_gamma = filter_gamma.T

    corr_matrices_delta = []
    corr_matrices_theta = []
    corr_matrices_alpha = []
    corr_matrices_beta = []
    corr_matrices_gamma = []

    for i in range(data.shape[0]):
        corr_matrix1 = np.zeros((channels, channels))
        corr_matrix2 = np.zeros((channels, channels))
        corr_matrix3 = np.zeros((channels, channels))
        corr_matrix4 = np.zeros((channels, channels))
        corr_matrix5 = np.zeros((channels, channels))
        rows1 = filter_delta[i * frequency:(i + 1) * frequency, :]
        rows2 = filter_theta[i * frequency:(i + 1) * frequency, :]
        rows3 = filter_alpha[i * frequency:(i + 1) * frequency, :]
        rows4 = filter_beta[i * frequency:(i + 1) * frequency, :]
        rows5 = filter_gamma[i * frequency:(i + 1) * frequency, :]
        for j in range(channels):
            for k in range(channels):
                corr_matrix1[j, k] = stats.pearsonr(rows1[:, j], rows1[:, k])[0]
                corr_matrix2[j, k] = stats.pearsonr(rows2[:, j], rows2[:, k])[0]
                corr_matrix3[j, k] = stats.pearsonr(rows3[:, j], rows3[:, k])[0]
                corr_matrix4[j, k] = stats.pearsonr(rows4[:, j], rows4[:, k])[0]
                corr_matrix5[j, k] = stats.pearsonr(rows5[:, j], rows5[:, k])[0]
        corr_matrices_delta.append(corr_matrix1)
        corr_matrices_theta.append(corr_matrix2)
        corr_matrices_alpha.append(corr_matrix3)
        corr_matrices_beta.append(corr_matrix4)
        corr_matrices_gamma.append(corr_matrix5)

    corr_matrices_delta = np.array(corr_matrices_delta)
    corr_matrices_theta = np.array(corr_matrices_theta)
    corr_matrices_alpha = np.array(corr_matrices_alpha)
    corr_matrices_beta = np.array(corr_matrices_beta)
    corr_matrices_gamma = np.array(corr_matrices_gamma)

    print(corr_matrices_delta.shape)
    print(corr_matrices_theta.shape)
    print(corr_matrices_alpha.shape)
    print(corr_matrices_beta.shape)
    print(corr_matrices_gamma.shape)

    Correlation = np.stack([corr_matrices_delta,corr_matrices_theta,corr_matrices_alpha,corr_matrices_beta,corr_matrices_gamma])
    # Correlation = Correlation.tolist()
    Correlation = Correlation.transpose(1,2,3,0)
    print(Correlation.shape)

    return Correlation

# if __name__ == '__main__':
#     # 加载每次选取30s视频后的原始数据文件
#     folder_path = 'E:/组合脑相关内容/SEED/第三个会话/raw'
#     # 获取指定文件夹下的所有.mat文件名,os.listdir()函数用于获取指定文件夹下的所有文件名
#     file_names = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
#     for file_name in file_names:
#         file_path = os.path.join(folder_path, file_name)
#         # mat_file = np.load(file_path)
#         print('processing {}'.format(file_path))
#         # 计算每个被试的皮尔逊相关系数
#         Pearson_feature = decompose_to_correlation(file_path)
#         print(Pearson_feature.shape)
#         #单被试结果
#         np.save("E:/组合脑相关内容/SEED/第三个会话/PPCC/{}.npy".format(file_name), Pearson_feature)

if __name__ == '__main__':
    # Fill in your SEED-VIG dataset path
    filePath = 'E:/博士成果/跟吴老师的第一篇文章/数据集/My_fatigue/'
    dataName = ['llq_renwu1.mat', 'llq_renwu2.mat', 'llq_renwu3.mat',
                'lxy_renwu1.mat', 'lxy_renwu2.mat', 'lxy_renwu3.mat',
                'yw_renwu1.mat', 'yw_renwu2.mat', 'yw_renwu3.mat',
                'xsw_renwu1.mat', 'xsw_renwu2.mat', 'xsw_renwu3.mat',
                'lsh_renwu1.mat', 'lsh_renwu2.mat', 'lsh_renwu3.mat',
                'lpr_renwu1.mat', 'lpr_renwu2.mat', 'lpr_renwu3.mat',
                'wmc_renwu1.mat', 'wmc_renwu2.mat', 'wmc_renwu3.mat',
                'pb_renwu1.mat', 'pb_renwu2.mat', 'pb_renwu3.mat',
                'tx_renwu1.mat', 'tx_renwu2.mat', 'tx_renwu3.mat',
                'wkj_renwu1.mat', 'wkj_renwu2.mat', 'wkj_renwu3.mat'
                # 'lxm_renwu1.mat', 'lxm_renwu2.mat', 'lxm_renwu3.mat',
                # 'llx_renwu1.mat', 'llx_renwu2.mat', 'llx_renwu3.mat',
                # 'jc_renwu1.mat', 'jc_renwu2.mat', 'jc_renwu3.mat'
                ]


    for i in range(len(dataName)):
        dataFile = filePath + dataName[i]
        print('processing {}'.format(dataName[i]))
        # every subject DE feature
        DE_feature = decompose_to_correlation(dataFile)
        np.save("E:/博士成果/NIPS论文/My_fatigue/PCC/{}.npy".format(i + 1), DE_feature)
