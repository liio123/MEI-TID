import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import warnings
from SEED_IV.cross_session_dataloader import *
from SEED_IV.Index_calculation import *
import matplotlib.pyplot as plt
from pylab import *
from SEED_IV.layers import GraphConvolution,Linear,GraphAttentionLayer
from SEED_IV.utils import *

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
        return (beta * z).sum(1), beta

class Basic(nn.Module):
    def __init__(self):
        super(Basic, self).__init__()
        self.BN2 = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.A = nn.Parameter(torch.FloatTensor(batch_size, 5, 10, 10).cuda())
        self.A = nn.init.kaiming_normal(self.A,)

        self.conv = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=10)
    def forward(self,fadj):

        fadj = fadj.permute(0, 3, 1, 2)

        # fadj = self.BN2(fadj)

        # sadj = self.BN2(self.A)

        # basic模块
        fadj = self.relu(F.softmax(self.conv(fadj), dim=1) * fadj)
        sadj = self.relu(F.softmax(self.conv(self.A), dim=1) * self.A)
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
        self.GCN1 = Chebynet(xdim, kadj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(5)
        self.basic = Basic()
        self.ract1 = E2R(epochs=epochs)
        self.relu = nn.ReLU()
        self.attentionadj = Attentionadj(10)

        self.w = nn.Parameter(torch.ones(4))

        self.hidden = nn.Sequential(
            nn.Conv2d(5,5,1,groups=5),
            nn.Linear(10,10),
            nn.Sigmoid()
        )
        self.essential = nn.Sequential(
            nn.Conv2d(5,5,1,groups=5),
            nn.Linear(10,10),
            nn.Sigmoid()
        )
        self.fc62 = nn.Sequential(
            # nn.BatchNorm1d(248),
            nn.Linear(40*16, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )
        self.fc4 = nn.Linear(40, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, fadj):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)

        x1 = x[:, 0:10, :]
        x2 = x[:, 50:60, :]
        x3 = x[:, 20:30, :]
        x4 = x[:, 10:20, :]

        fadj1 = fadj[:, 0:10, 0:10, :]
        fadj2 = fadj[:, 50:60, 50:60, :]
        fadj3 = fadj[:, 20:30, 20:30, :]
        fadj4 = fadj[:, 10:20, 10:20, :]

        #basic模块
        fadj1, sadj1 = self.basic(fadj1)
        fadj2, sadj2 = self.basic(fadj2)
        fadj3, sadj3 = self.basic(fadj3)
        fadj4, sadj4 = self.basic(fadj4)
        # print(fadj1)

        #转为黎曼流形空间
        fadj1 = self.ract1(fadj1)
        sadj1 = self.ract1(sadj1)
        fadj2 = self.ract1(fadj2)
        sadj2 = self.ract1(sadj2)
        fadj3 = self.ract1(fadj3)
        sadj3 = self.ract1(sadj3)
        fadj4 = self.ract1(fadj4)
        sadj4 = self.ract1(sadj4)
        # x = self.ract1(x)
        # print(fadj1)

        #从原邻接矩阵和噪声矩阵抽象出本质连接和隐藏连接
        com1 = (fadj1 + sadj1) / 2
        com2 = (fadj2 + sadj2) / 2
        com3 = (fadj3 + sadj3) / 2
        com4 = (fadj4 + sadj4) / 2
        # print(com1)
        hidden1 = self.hidden(com1.cuda())
        essential1 = self.essential(com1.cuda())
        hidden2 = self.hidden(com2.cuda())
        essential2 = self.essential(com2.cuda())
        hidden3 = self.hidden(com3.cuda())
        essential3 = self.essential(com3.cuda())
        hidden4 = self.hidden(com4.cuda())
        essential4 = self.essential(com4.cuda())
        # print(hidden1)

        hidden1, att = self.attentionadj(hidden1)
        essential1, att = self.attentionadj(essential1)
        hidden2, att = self.attentionadj(hidden2)
        essential2, att = self.attentionadj(essential2)
        hidden3, att = self.attentionadj(hidden3)
        essential3, att = self.attentionadj(essential3)
        hidden4, att = self.attentionadj(hidden4)
        essential4, att = self.attentionadj(essential4)
        # print(hidden1)
        # hidden1 = normalize_A(hidden1, symmetry=False, gaowei=True)
        # essential1 = normalize_A(essential1, symmetry=False, gaowei=False)
        # hidden2 = normalize_A(hidden2, symmetry=False, gaowei=True)
        # essential2 = normalize_A(essential2, symmetry=False, gaowei=False)
        # hidden3 = normalize_A(hidden3, symmetry=False, gaowei=True)
        # essential3 = normalize_A(essential3, symmetry=False, gaowei=False)
        #在黎曼流形空间和欧几里得空间的图卷积计算
        # print(x1.shape)
        h1 = self.relu(self.GCN1(x1, hidden1))
        e1 = self.relu(self.GCN1(x1, essential1))
        h2 = self.relu(self.GCN1(x2, hidden2))
        e2 = self.relu(self.GCN1(x2, essential2))
        h3 = self.relu(self.GCN1(x3, hidden3))
        e3 = self.relu(self.GCN1(x3, essential3))
        h4 = self.relu(self.GCN1(x4, hidden4))
        e4 = self.relu(self.GCN1(x4, essential4))
        # print(h1.shape)

        feature1 = (h1 + e1) / 2
        feature2 = (h2 + e2) / 2
        feature3 = (h3 + e3) / 2
        feature4 = (h4 + e4) / 2
        # print(feature1)

        #多域注意力机制
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))

        # feature = feature1 * w1 + feature2 * w2 + feature3 * w3 + feature4 * w4
        feature = torch.cat([feature1 * w1, feature2 * w2, feature3 * w3, feature4 * w4], dim=1)
        # print(feature)
        feature = self.relu(feature)

        output = feature.view(batch_size, -1)
        output = self.fc62(output)
        output = self.fc4(output)

        return output

warnings.filterwarnings("ignore")

learning_rate = 0.008
epochs = 5
epoch = 200
min_acc = 0.7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myModel = model(xdim = [256,62,5],kadj=2,num_out=16,dropout=0.5).to(device)
loss_func = nn.MSELoss()
loss_feature = nn.NLLLoss()
loss_cross = nn.CrossEntropyLoss()
# loss_common = common_loss()
# opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate, weight_decay=0)
opt = optim.SGD(myModel.parameters(), lr=0.01, momentum=0.5)

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
        x, fadj, y = data
        x = x.to(device)
        fadj = fadj.to(device)
        y = y.to(device)
        output = myModel(x,fadj)
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
            x, fadj, y = data
            x = x.to(device)
            fadj = fadj.to(device)
            y = y.to(device)
            output = myModel(x,fadj)
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
    #     torch.save(myModel.state_dict(), 'D:/SEED/三个损失函数的模型保存/S5.pkl')


    print("Epoch: {}/{} ".format(i + 1, epoch),
          "Training Loss: {:.4f} ".format(total_train_loss / len(train_dataloader)),
          "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
          "Test Loss: {:.4f} ".format(total_test_loss / len(test_dataloader)),
          "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
          )


print(min_acc)
# print("TP: {}".format(res_TP_TN_FP_FN[0]))
# print("TN: {}".format(res_TP_TN_FP_FN[1]))
# print("FP: {}".format(res_TP_TN_FP_FN[2]))
# print("FN: {}".format(res_TP_TN_FP_FN[3]))

train_x1 = range(0, epoch)
train_x2 = range(0, epoch)
train_y1 = Train_Accuracy_list
train_y2 = Train_Loss_list
plt.subplot(2, 1, 1)
plt.plot(train_x1, train_y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(train_x2, train_y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.show()
#
test_x1 = range(0, 200)
test_x2 = range(0, 200)
test_y1 = Test_Accuracy_list
test_y2 = Test_Loss_list
plt.subplot(2, 1, 1)
plt.plot(test_x1, test_y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(test_x2, test_y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()