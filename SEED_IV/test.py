import numpy as np
from scipy.io import loadmat

DE = np.load("E:/博士成果/NIPS论文/SEED-IV/DE/1/1_20160518.mat.npy")
PCC = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/1/1_20160518.mat.npy.npy")

print(DE.shape, PCC.shape)

