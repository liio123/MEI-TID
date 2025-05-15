import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import warnings
from SEED.cross_session_dataloader import *
from SEED.Index_calculation import *
import matplotlib.pyplot as plt
from pylab import *
from SEED.layers import GraphConvolution,Linear,GraphAttentionLayer
from SEED.utils import *

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
        self.fc4 = nn.Linear(all_channel_num, 3)
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

            # 转为黎曼流形空间
            fadj = self.ract1(fadj)
            sadj = self.ract1(sadj)

            # 从原邻接矩阵和噪声矩阵抽象出本质连接和隐藏连接
            com = (fadj + sadj) / 2
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
        output = self.fc62(output)
        output = self.fc4(output)

        return output

warnings.filterwarnings("ignore")

learning_rate = 0.008
epochs = 5
epoch = 200
min_acc = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myModel = model(xdim = [batch_size, all_channel_num, band_num], kadj=2, num_out=16, dropout=0.5).to(device)
loss_func = nn.MSELoss()
loss_feature = nn.NLLLoss()
loss_cross = nn.CrossEntropyLoss()
# loss_common = common_loss()
opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate, weight_decay=0.05)
# opt = optim.SGD(myModel.parameters(), lr=0.01, momentum=0.5)

G = testclass()
train_len = G.len(X_train1.shape[0], batch_size)
test_len = G.len(X_test1.shape[0], batch_size)


train_loss_plt = []
train_acc_plt = []
test_loss_plt = []
test_acc_plt = []
Train_Loss_list = []
Train_Accuracy_list = []
Test_Loss_list = []
Test_Accuracy_list = []

for i in range(epoch):
    total_train_step = 0
    total_test_step = 0

    total_train_loss = 0
    total_train_acc = 0


    for data in train_dataloader:
        list_len = len(data)
        brain_region_num = int((list_len - 1) / 2)
        de_list = [data[i].to(device) for i in range(0, brain_region_num)]
        pcc_list = [data[i].to(device) for i in range(brain_region_num, list_len-1)]
        y = data[list_len-1].to(device)

        output = myModel(de_list, pcc_list)
        # print(output5, y)
        train_loss_task = loss_cross(output, y)


        opt.zero_grad()
        train_loss_task.backward()
        opt.step()

        # print(output5)

        train_acc = (output.argmax(dim=1)  == y).sum()
        # train_acc = G.acc(output5, y)
        # print(train_acc)

        train_loss_plt.append(train_loss_task)
        total_train_loss = total_train_loss + train_loss_task.item()
        total_train_step = total_train_step + 1

        train_acc_plt.append(train_acc)
        total_train_acc += train_acc

    Train_Loss_list.append(total_train_loss / (len(train_dataloader)))
    Train_Accuracy_list.append(total_train_acc / train_len)

    total_test_loss = 0
    total_test_acc = 0
    matrix = [0, 0, 0, 0]


    with torch.no_grad():
        pred_output_list = []

        for data in test_dataloader:
            list_len = len(data)
            brain_region_num = int((list_len - 1) / 2)
            de_list = [data[i].to(device) for i in range(0, brain_region_num)]
            pcc_list = [data[i].to(device) for i in range(brain_region_num, list_len - 1)]
            y = data[list_len - 1].to(device)
            output = myModel(de_list, pcc_list)
            test_loss_task = loss_cross(output, y)

            test_acc = (output.argmax(dim=1) == y).sum()
            # TP_TN_FP_FN = G.Compute_TP_TN_FP_FN(test_label, label, matrix)

            test_loss_plt.append(test_loss_task)
            total_test_loss = total_test_loss + test_loss_task.item()
            total_test_step = total_test_step + 1

            test_acc_plt.append(test_acc)
            total_test_acc += test_acc


    Test_Loss_list.append(total_test_loss / (len(train_dataloader)))
    Test_Accuracy_list.append(total_test_acc / train_len)
    #
    if(total_test_acc / test_len) > min_acc:
        min_acc = total_test_acc / test_len
    #     res_TP_TN_FP_FN = TP_TN_FP_FN
        torch.save(myModel.state_dict(), 'E:/博士成果/NIPS论文/SEED/跨会话结果/12_3/S{}.pkl'.format(k))


    print("Epoch: {}/{} ".format(i + 1, epoch),
          "Training Loss: {:.4f} ".format(total_train_loss / len(train_dataloader)),
          "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
          "Test Loss: {:.4f} ".format(total_test_loss / len(test_dataloader)),
          "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
          )


print(min_acc)
