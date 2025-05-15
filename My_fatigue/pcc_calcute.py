import numpy as np
from scipy.signal import butter, lfilter
import os
from scipy.io import loadmat


#  假设采样频率为400hz,信号本身最大的频率为200hz，要滤除0.5hz以下，50hz以上频率成分，即截至频率为0.5hz，50hz
def butter_bandpass_filter(data, lowcut, highcut, samplingRate, order=5):
	nyq = 0.5 * samplingRate
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	y = lfilter(b, a, data)
	return y


# -------------------------------------------------------------------------
# # calculate pcc
def decompose_to_correlation(file_path):
    frequency = 1000
    data = loadmat(file_path)["EEG"]["data"][0][0]  # (31, 9429560)
    data = np.concatenate((data[:, :100*frequency], data[:, -100*frequency:]), axis=1)  # (31, 200000)
    print(data.shape)
    channels = data.shape[0]

    data_list = []
    for channel in range(channels):

        trail_single = data[channel, :]

        Delta = butter_bandpass_filter(trail_single, 1, 4, frequency, order=3)
        Theta = butter_bandpass_filter(trail_single, 4, 8, frequency, order=3)
        Alpha = butter_bandpass_filter(trail_single, 8, 14, frequency, order=3)
        Beta = butter_bandpass_filter(trail_single, 14, 30, frequency, order=3)
        Gamma = butter_bandpass_filter(trail_single, 30, 48, frequency, order=3)

        Fre_list = [Delta, Theta, Alpha, Beta, Gamma]
        data_list.append(Fre_list)

    data = np.array(data_list).transpose(2, 0, 1)   # (9508740, 31, 5)
    vsplit_data = np.vsplit(data, 200) # (1000, 200, 31, 5)

    cor_list = []
    for vsplit_datum in vsplit_data:
        vsplit_datum = vsplit_datum.transpose(2, 1, 0)  # (5, 31, 1000)

        f_list = []
        for c in range(5):
            cor = np.corrcoef(vsplit_datum[c, :, :])
            f_list.append(cor)
        cor_list.append(f_list)

    corr = np.array(cor_list)
    return corr


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
        Pearson_feature = decompose_to_correlation(file_path)
        print(Pearson_feature.shape)
        #单被试结果
        # np.save(f"E:/博士成果/NIPS论文/My_fatigue/PCC/3/{i}.npy", Pearson_feature)
        i = i + 1