import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

k=2
S = np.load("E:博士成果/NIPS论文/SEED-IV/Structure/S{}.npy".format(k), allow_pickle=True).item()

DE1 = np.load("E:/博士成果/NIPS论文/SEED-IV/DE/1/{}.npy".format(k))
DE2 = np.load("E:/博士成果/NIPS论文/SEED-IV/DE/2/{}.npy".format(k))
DE3 = np.load("E:/博士成果/NIPS论文/SEED-IV/DE/3/{}.npy".format(k))

PCC1 = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/1/{}.npy".format(k))
PCC2 = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/2/{}.npy".format(k))
PCC3 = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/3/{}.npy".format(k))

# print(DE1.shape, DE2.shape, DE3.shape)
# print(PCC1.shape, PCC2.shape, PCC3.shape)

session1 = np.concatenate([np.ones([42, 1]), np.full((23,1), 2), np.full((49,1), 3), np.zeros([32, 1]), np.full((22,1), 2), np.zeros([40, 1]), np.zeros([38, 1]), np.ones([52, 1]),
                          np.zeros([36, 1]), np.ones([42, 1]), np.full((12,1), 2), np.ones([27, 1]), np.ones([54, 1]), np.ones([42, 1]), np.full((64,1), 2), np.full((35,1), 3),
                          np.full((17,1), 2), np.full((44,1), 2), np.full((35,1), 3), np.full((12,1), 3), np.zeros([28, 1]), np.full((28,1), 3), np.zeros([43, 1]), np.full((34,1), 3)])

session2 = np.concatenate([np.full((55,1), 2), np.ones([25, 1]), np.full((34,1), 3), np.zeros([36, 1]), np.zeros([53, 1]), np.full((27,1), 2), np.zeros([34, 1]), np.full((46,1), 2),
                          np.full((34,1), 3), np.full((20,1), 3), np.full((60,1), 2), np.full((12,1), 3), np.full((36,1), 2), np.zeros([27, 1]), np.ones([44, 1]), np.ones([15, 1]),
                          np.full((46,1), 2), np.ones([49, 1]), np.zeros([45, 1]), np.full((10,1), 3), np.zeros([37, 1]), np.ones([44, 1]), np.full((24,1), 3), np.ones([19, 1])])

session3 = np.concatenate([np.ones([42, 1]), np.full((32,1), 2), np.full((23,1), 2), np.ones([45, 1]), np.full((48,1), 3), np.full((26,1), 3), np.full((64,1), 3), np.ones([23, 1]),
                          np.ones([26, 1]), np.full((16,1), 2), np.ones([51, 1]), np.zeros([41, 1]), np.full((39,1), 2), np.full((19,1), 3), np.full((28,1), 3), np.zeros([44, 1]),
                          np.full((14,1), 2), np.full((17,1), 3), np.zeros([45, 1]), np.zeros([22, 1]), np.full((39,1), 2), np.zeros([38, 1]), np.ones([41, 1]), np.zeros([39, 1])])
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
batch_size = 800
trainData = TensorDataset(*train_DE_list, *train_PCC_list, Y_train)
testData = TensorDataset(*test_DE_list, *test_PCC_list, Y_test)
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")