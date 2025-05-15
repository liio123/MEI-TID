import numpy as np
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

k=1
S = np.load("G:博士成果/NIPS论文/My_fatigue/Structure/S{}.npy".format(k), allow_pickle=True).item()

DE1 = np.load("G:/博士成果/NIPS论文/My_fatigue/DE/1/{}.npy".format(k))
DE2 = np.load("G:/博士成果/NIPS论文/My_fatigue/DE/2/{}.npy".format(k))
DE3 = np.load("G:/博士成果/NIPS论文/My_fatigue/DE/3/{}.npy".format(k))

PCC1 = np.load("G:/博士成果/NIPS论文/My_fatigue/PCC/1/{}.npy".format(k))
PCC2 = np.load("G:/博士成果/NIPS论文/My_fatigue/PCC/2/{}.npy".format(k))
PCC3 = np.load("G:/博士成果/NIPS论文/My_fatigue/PCC/3/{}.npy".format(k))

PCC1 = PCC1.transpose(0,2,3,1)
PCC2 = PCC2.transpose(0,2,3,1)
PCC3 = PCC3.transpose(0,2,3,1)
# print(DE1.shape, DE2.shape, DE3.shape)
# print(PCC1.shape, PCC2.shape, PCC3.shape)

session1 = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])])

session2 = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])])

session3 = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])])
# print(session1.shape, session2.shape, session3.shape)
X_train1 = np.concatenate([DE1, DE2])   # (1683, 62, 5)
X_train2 = np.concatenate([PCC1, PCC2]).transpose(0, 3, 1, 2)     # (1683, 62, 62, 5)
X_test1 = DE3
X_test2 = PCC3.transpose(0, 3, 1, 2)

Y_train = np.concatenate([session1, session2])
Y_test = session3   # (822, 1)

train_DE_list = []
train_PCC_list = []
test_DE_list = []
test_PCC_list = []
value_num_list = []
for key, value in S.items():
    locals()[f'train_DE_{key}'] = torch.tensor(np.stack([X_train1[:, i, :] for i in value]).transpose(1, 0, 2), dtype=torch.float)  # torch.Size([1683, 9, 5])
    locals()[f'train_PCC_{key}'] = torch.tensor(X_train2[:, :, value][:, :, :, value], dtype=torch.float).permute(0, 2, 3, 1)       # torch.Size([1683, 9, 9, 5])
    locals()[f'test_DE_{key}'] = torch.tensor(np.stack([X_test1[:, i, :] for i in value]).transpose(1, 0, 2), dtype=torch.float)
    locals()[f'test_PCC_{key}'] = torch.tensor(X_test2[:, :, value][:, :, :, value], dtype=torch.float).permute(0, 2, 3, 1)

    train_DE_list.append(locals()[f'train_DE_{key}'])
    train_PCC_list.append(locals()[f'train_PCC_{key}'])
    test_DE_list.append(locals()[f'test_DE_{key}'])
    test_PCC_list.append(locals()[f'test_PCC_{key}'])

    value_num_list.append(len(value))
value_num_len = len(value_num_list)
channel_num_list = list(set(value_num_list))

Y_train = torch.tensor(Y_train, dtype=torch.int64).squeeze_(1)
Y_test = torch.tensor(Y_test, dtype=torch.int64).squeeze_(1)

print("训练集测试集已划分完成............")
batch_size = 50
trainData = TensorDataset(*train_DE_list, *train_PCC_list, Y_train)
testData = TensorDataset(*test_DE_list, *test_PCC_list, Y_test)
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")