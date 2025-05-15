import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import random

DE = np.load("E:/博士成果/NIPS论文/SEED-IV/DE/1/2_20150915.mat.npy")
PCC = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/1/2_20150915.mat.npy.npy")

X_train1 = DE[0:610,:,:]
X_train2 = PCC[0:610,:,:,:]
X_test1 = DE[610:851,:,:]
X_test2 = PCC[610:851,:,:,:]
Y_train = np.concatenate([np.ones([42, 1]), np.full((23,1), 2), np.full((49,1), 3), np.zeros([32, 1]), np.full((22,1), 2), np.zeros([40, 1]), np.zeros([38, 1]), np.ones([52, 1]),
                          np.zeros([36, 1]), np.ones([42, 1]), np.full((12,1), 2), np.ones([27, 1]), np.ones([54, 1]), np.ones([42, 1]), np.full((64,1), 2), np.full((35,1), 3)])
Y_test = np.concatenate([np.full((17,1), 2), np.full((44,1), 2), np.full((35,1), 3), np.full((12,1), 3), np.zeros([28, 1]), np.full((28,1), 3), np.zeros([43, 1]), np.full((34,1), 3)])

X_train1 = torch.tensor(X_train1, dtype=torch.float)
X_train2 = torch.tensor(X_train2, dtype=torch.float)
X_test1 = torch.tensor(X_test1, dtype=torch.float)
X_test2 = torch.tensor(X_test2, dtype=torch.float)
Y_train = torch.tensor(Y_train, dtype=torch.int64).squeeze_(1)
Y_test = torch.tensor(Y_test, dtype=torch.int64).squeeze_(1)


print("训练集测试集已划分完成............")
batch_size = 64
trainData = TensorDataset(X_train1, X_train2, Y_train)
testData = TensorDataset(X_test1, X_test2, Y_test)
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")