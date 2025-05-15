import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import warnings
from My_fatigue.cross_session_dataloader import *
from My_fatigue.Index_calculation import *
import matplotlib.pyplot as plt
from pylab import *
from My_fatigue.layers import GraphConvolution,Linear,GraphAttentionLayer
from My_fatigue.utils import *
import mne

band_num = DE1.shape[2]
all_channel_num = DE1.shape[1]

class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        self.dp = nn.Dropout(dropout)
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                # print(result.shape)
                # print(x.shape)
                # print(adj[i].shape)
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        #result = self.dp(result)
        return result

class Attentionadj(nn.Module):
    def __init__(self, in_size, hidden_size=10):
        super(Attentionadj, self).__init__()

        self.project = nn.Sequential(
            #nn.BatchNorm2d(3),
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)

class Basic(nn.Module):
    def __init__(self):
        super(Basic, self).__init__()
        self.BN2 = nn.BatchNorm2d(band_num)
        self.relu = nn.ReLU()
        self.A_dict = {}
        self.conv_dict = {}
        for channel_num in channel_num_list:
            self.A_dict[f'A{channel_num}'] = nn.Parameter(torch.FloatTensor(batch_size, band_num, channel_num, channel_num).cuda()).to(device)
            self.A_dict[f'A{channel_num}'] = nn.init.kaiming_normal(self.A_dict[f'A{channel_num}'],).to(device)
            self.conv_dict[f'conv{channel_num}'] = nn.Conv2d(in_channels=band_num, out_channels=band_num, kernel_size=channel_num).to(device)

    def forward(self,fadj, c_num):

        fadj = fadj.permute(0, 3, 1, 2)

        # fadj = self.BN2(fadj)

        # sadj = self.BN2(self.A)

        # basic模块
        fadj = self.relu(F.softmax(self.conv_dict[f'conv{c_num}'](fadj), dim=1) * fadj)
        sadj = self.relu(F.softmax(self.conv_dict[f'conv{c_num}'](self.A_dict[f'A{c_num}']), dim=1) * self.A_dict[f'A{c_num}'])
        # print(fadj)
        return fadj, sadj

class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')

    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        # print(x.shape)
        cov = x.permute(0, 2, 1) @ x
        # print(cov.shape)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        # print(cov.shape)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        # print(tra.shape)
        tra = tra.view(-1, 1, 1)
        # print(tra.shape)
        cov /= tra
        # print(cov.shape)
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        # print(identity.shape)
        cov = cov + (1e-5 * identity)
        # print(cov.shape)
        return cov

class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

    def patch_len(self, n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        # x with shape[bs, ch, time]

        list_patch = self.patch_len(x.shape[1], int(self.epochs))
        # print("list_patch:", list_patch)
        x_list = list(torch.split(x, list_patch, dim=1))
        # print("x_list:", x_list)
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x



class model(nn.Module):
    def __init__(self, xdim, kadj, num_out, dropout):
        super(model, self).__init__()
        self.GCN = Chebynet(xdim, kadj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(5)
        self.basic = Basic()
        self.ract1 = E2R(epochs=epochs)
        self.relu = nn.ReLU()
        self.attentionadj_dict = {}
        for channel_num in channel_num_list:
            self.attentionadj_dict[f'attentionadj{channel_num}'] = Attentionadj(channel_num).to(device)

        self.w = nn.Parameter(torch.ones(value_num_len))

        self.hidden_dict = {}
        for channel_num in channel_num_list:
            self.hidden_dict[f'hidden{channel_num}'] = nn.Sequential(
            nn.Conv2d(band_num,band_num, 1, groups=band_num),
            nn.Linear(channel_num, channel_num),
            nn.Sigmoid()
        ).to(device)

        self.essential_dict = {}
        for channel_num in channel_num_list:
            self.essential_dict[f'essential{channel_num}'] = nn.Sequential(
                nn.Conv2d(band_num,band_num, 1, groups=band_num),
                nn.Linear(channel_num, channel_num),
                nn.Sigmoid()
            ).to(device)

        self.fc62 = nn.Sequential(
            # nn.BatchNorm1d(248),
            nn.Linear(all_channel_num*16, all_channel_num),
            nn.BatchNorm1d(all_channel_num),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )
        self.fc4 = nn.Linear(all_channel_num, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, de_list, pcc_list):
        new_de_list = []
        for de in de_list:
            de = self.BN1(de.transpose(1, 2)).transpose(1, 2)
            new_de_list.append(de)

        hidden_list = []
        essential_list = []
        for pcc in pcc_list:
            c_num = pcc.shape[2]
            # basic模块
            fadj, sadj = self.basic(pcc, c_num)
            # print(fadj.shape, sadj.shape)

            # 转为黎曼流形空间
            fadj = self.ract1(fadj)
            sadj = self.ract1(sadj)
            # print(fadj.shape, sadj.shape)

            # 从原邻接矩阵和噪声矩阵抽象出本质连接和隐藏连接
            com = (fadj + sadj) / 2
            # print(com.shape)
            # print(c_num)
            hidden = self.hidden_dict[f'hidden{c_num}'](com.cuda())
            essential = self.essential_dict[f'essential{c_num}'](com.cuda())

            hidden = self.attentionadj_dict[f'attentionadj{c_num}'](hidden)
            essential = self.attentionadj_dict[f'attentionadj{c_num}'](essential)

            hidden_list.append(hidden)
            essential_list.append(essential)

        feature_list = []
        #在黎曼流形空间和欧几里得空间的图卷积计算
        for i in range(len(de_list)):
            h = self.relu(self.GCN(de_list[i], hidden_list[i]))
            e = self.relu(self.GCN(de_list[i], essential_list[i]))
            feature = (h + e) / 2
            # print(feature.shape)
            # 多域注意力机制
            w = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
            feature_list.append(feature * w)
        feature = torch.cat(feature_list, dim=1)
        # print(feature.shape)
        feature = self.relu(feature)
        output = feature.view(batch_size, -1)
        output1 = self.fc62(output)
        output = self.fc4(output1)

        return output1, output



k = 2
batch_size = 150


S = np.load(f"E:/博士成果/NIPS论文/My_fatigue/Structure/S{k}.npy", allow_pickle=True).item()

DE2 = np.load(f"E:/博士成果/NIPS论文/My_fatigue/DE/2/{k}.npy")
band_num = DE2.shape[2]
all_channel_num = DE2.shape[1]
PCC2 = np.load(f"E:/博士成果/NIPS论文/My_fatigue/PCC/2/{k}.npy").transpose(0, 2, 3, 1).transpose(0, 3, 1, 2)

session1 = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])])
session2 = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])])
session3 = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])])

X_test1 = DE2
X_test2 = PCC2

Y_test = session3   # (822, 1)

print(DE2.shape)

train_DE_list = []
train_PCC_list = []
test_DE_list = []
test_PCC_list = []
value_num_list = []
for key, value in S.items():
    locals()[f'test_DE_{key}'] = torch.tensor(np.stack([X_test1[:, i, :] for i in value]).transpose(1, 0, 2), dtype=torch.float)
    locals()[f'test_PCC_{key}'] = torch.tensor(X_test2[:, :, value][:, :, :, value], dtype=torch.float).permute(0, 2, 3, 1)
    # print(locals()[f'test_DE_{key}'].shape)
    # print(locals()[f'test_PCC_{key}'].shape)

    test_DE_list.append(locals()[f'test_DE_{key}'])
    test_PCC_list.append(locals()[f'test_PCC_{key}'])

    value_num_list.append(len(value))
value_num_len = len(value_num_list)
channel_num_list = list(set(value_num_list))

Y_test = torch.tensor(Y_test, dtype=torch.int64).squeeze_(1)

print("测试集已划分完成............")
testData = TensorDataset(*test_DE_list, *test_PCC_list, Y_test)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")

band_num = DE2.shape[2]
all_channel_num = DE2.shape[1]

warnings.filterwarnings("ignore")

epochs = 5
min_acc = 0.7
epoch = 200
fold = "13_2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myModel = model(xdim = [batch_size, all_channel_num, band_num], kadj=2, num_out=16, dropout=0.5).to(device)
myModel.load_state_dict(torch.load(f"E:/博士成果/NIPS论文/My_fatigue/跨会话结果/13_2/S{k}.pkl"))
loss_func = nn.MSELoss()
loss_feature = nn.NLLLoss()
loss_cross = nn.CrossEntropyLoss()
# loss_common = common_loss()
opt = torch.optim.Adam(myModel.parameters(), lr=0.008, weight_decay=0.001)
# opt = optim.SGD(myModel.parameters(), lr=0.01, momentum=0.5)

G = testclass()
test_len = G.len(X_test1.shape[0], batch_size)

test_loss_plt = []
test_acc_plt = []
Test_Loss_list = []
Test_Accuracy_list = []

total_test_step = 0
total_test_acc = 0


pred_output_list = []

for data in test_dataloader:
    list_len = len(data)
    brain_region_num = int((list_len - 1) / 2)
    de_list = [data[i].to(device) for i in range(0, brain_region_num)]
    pcc_list = [data[i].to(device) for i in range(brain_region_num, list_len - 1)]
    y = data[list_len - 1].to(device)
    output1, output = myModel(de_list, pcc_list)

    # test_acc = (output.argmax(dim=1) == y).sum()
    # # TP_TN_FP_FN = G.Compute_TP_TN_FP_FN(test_label, label, matrix)
    # total_test_step = total_test_step + 1

    # test_acc_plt.append(test_acc)
    # total_test_acc += test_acc

# Test_Accuracy_list.append(total_test_acc / test_len)
#
# if(total_test_acc / test_len) > min_acc:
#     min_acc = total_test_acc / test_len
# #     res_TP_TN_FP_FN = TP_TN_FP_FN
#     torch.save(myModel.state_dict(), f'E:/SEED-IV/Model/{fold}/S{k}.pkl')


# print("Epoch: {}/{} ".format(i + 1, epoch),
#       "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
#       )


# print(min_acc)

output62 = output1.cpu().detach().numpy()
output62 = np.array(output62)
print(output62.shape)
# 读取导联位置信息，创建对应的info

# 读取MNE中biosemi电极位置信息
# biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# print(biosemi_montage.get_positions())
# sensor_data = biosemi_montage.get_positions()['ch_pos']
# print(sensor_data)
# sensor_dataframe = pd.DataFrame(sensor_data).T
# print(sensor_dataframe)
# sensor_dataframe.to_excel('sensor_dataframe.xlsx')

# 获取的除ch_pos外的信息
'''
'coord_frame': 'unknown', 'nasion': array([ 5.27205792e-18,  8.60992398e-02, -4.01487349e-02]),
'lpa': array([-0.08609924, -0.        , -0.04014873]), 'rpa': array([ 0.08609924,  0.        , -0.04014873]),
'hsp': None, 'hpi': None
'''

# 将获取的电极位置信息修改并补充缺失的电极位置，整合为1020.xlsx
# data1020 = pd.read_excel('1020.xlsx', index_col=0)
# channels1020 = np.array(data1020.index)
# value1020 = np.array(data1020)
#
# # 将电极通道名称和对应三维坐标位置存储为字典形式
# list_dic = dict(zip(channels1020, value1020))
# # print(list_dic)
# # 封装为MNE的格式，参考原biosemi的存储格式
# montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
#                                              nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
#                                              lpa=[-0.08609924, -0., -0.04014873],
#                                              rpa=[0.08609924, 0., -0.04014873])

# 图示电极位置
# montage_1020.plot()
# plt.show()

# montage = mne.channels.read_custom_montage(montage_1020)
# print(montage)
montage_1020 = mne.channels.read_custom_montage('31.locs')
info = mne.create_info(ch_names=montage_1020.ch_names, sfreq=200, ch_types='eeg')

# 画图
# fig, ax = plt.subplots(ncols=3, figsize=(4, 4), gridspec_kw=dict(top=0.9), sharex=True, sharey=True)
# fig = plt.figure()
# fig.patch.set_facecolor('blue')
# # fig.patch.set_alpha(0)
# ax1 = fig.add_axes([0.5,0.5,0.5,0.5])
# plt.figure(figsize=(10, 10))
for i in range(150):

    # plt.subplot(1, 1, i + 1)
    evoked = mne.EvokedArray(output62[i,:].reshape(31, 1), info)
    # print(eeg_re_prm[:, i].reshape(31, 1))
    # print()
    evoked.set_montage(montage_1020)
    im, cm = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=False, cmap='viridis')

    plt.title("")
    # 添加所有子图的colorbar，
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

    # name = os.path.splitext(eeg_re_prm)[0]
    # plt.savefig(f"F:/疲劳数据/mne_map/{i}.svg")

    plt.close()

