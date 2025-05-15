import numpy as np
from sklearn.decomposition import PCA
import torch
from sklearn import cluster
import os
import matplotlib.pyplot as plt
from itertools import cycle

np.set_printoptions(threshold=np.inf)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def sby_dim_re(pcc):
    pca1 = PCA(n_components=1)
    pca2 = PCA(n_components=1)

    pcc_3 = pcc.reshape(pcc.shape[0], -1).transpose(1, 0)
    pca1.fit(pcc_3)
    pcc_3 = pca1.fit_transform(pcc_3)
    # print(pcc_3.shape)

    pcc_2 = pcc_3.reshape(pcc.shape[1], -1).transpose(1, 0)
    pca2.fit(pcc_2)
    pcc_2 = pca2.fit_transform(pcc_2)
    # print(pcc_2.shape)

    pcc_1 = pcc_2.reshape(pcc.shape[2], -1)
    # print(pcc_1.shape)

    return pcc_1

# convert signal epoch to SPD matrix
def signal2spd(x):
    x = x.squeeze()
    mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
    x = x - mean
    # print(x.shape)
    cov = x.permute(0, 2, 1) @ x
    # print(cov.shape)
    # cov = cov.to(self.dev)
    cov = cov / (x.shape[-1] - 1)
    # print(cov.shape)
    tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    # print(tra.shape)
    tra = tra.view(-1, 1, 1)
    # print(tra.shape)
    cov /= tra
    # print(cov.shape)
    identity = torch.eye(cov.shape[-1], cov.shape[-1]).repeat(x.shape[0], 1, 1)
    # print(identity.shape)
    cov = cov + (1e-5 * identity)
    # print(cov.shape)
    return cov

def patch_len(n, epochs):
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

def E2R(x, epochs):
    # x with shape[bs, ch, time]
    # list_patch = patch_len(x.shape[1], int(epochs))
    # print("list_patch:", list_patch)
    # x_list = list(torch.split(x, list_patch, dim=1))
    # print("x_list:", x_list)
    # for i, item in enumerate(x_list):
    #     x_list[i] = signal2spd(item)
    # x = torch.stack(x_list).permute(1, 0, 2, 3)
    x = signal2spd(torch.tensor(x))
    return x.reshape(x.shape[1], x.shape[2])

def find_indexes(array):
    index_dict = {}
    for i, num in enumerate(array):
        if num not in index_dict:
            index_dict[num] = [i]
        else:
            index_dict[num].append(i)
    return index_dict

def riemannian_cluster(Xn):
    ap = cluster.AffinityPropagation(damping=0.5, max_iter=500, convergence_iter=300, affinity='euclidean').fit(Xn)
    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    # new_X = np.column_stack((Xn, labels))

    n_clusters_ = len(cluster_centers_indices)
    # print(len(labels))
    # print("-----------------")
    # print(type(labels))
# ---------------------------------------------------------------
#     plt.close('all')
#     plt.figure(1)
#     plt.clf()
#
#     colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#     for k, col in zip(range(n_clusters_), colors):
#         class_members = labels == k
#         cluster_center = Xn[cluster_centers_indices[k]]
#         plt.plot(Xn[class_members, 0], Xn[class_members, 1], col + '.')
#         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#                  markeredgecolor='k', markersize=14)
#         for x in Xn[class_members]:
#             plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
#     plt.title('Estimated number of clusters: %d' % n_clusters_)
#     plt.show()
# ---------------------------------------------------------------
    labels = labels.tolist()
    index_dict = find_indexes(labels)

    return index_dict

k=15
PCC1 = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/1/{}.npy".format(k)).transpose(0, 3, 1, 2)      # (851, 5, 62, 62)
PCC2 = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/2/{}.npy".format(k)).transpose(0, 3, 1, 2)
PCC3 = np.load("E:/博士成果/NIPS论文/SEED-IV/PCC/3/{}.npy".format(k)).transpose(0, 3, 1, 2)

pcc_1 = sby_dim_re(PCC1)
pcc_2 = sby_dim_re(PCC2)
pcc_3 = sby_dim_re(PCC3)

stacked_matrix = np.dstack((pcc_1, pcc_2, pcc_3))   # (62, 62, 3)

flattened_matrix = stacked_matrix.reshape(-1, stacked_matrix.shape[-1])     # (3844, 3)

pca = PCA(n_components=1)
pca.fit(flattened_matrix)

combined_matrix = pca.fit_transform(flattened_matrix).reshape(stacked_matrix.shape[0], -1)

epochs = 5
niman = E2R(combined_matrix, epochs)
X = np.array(niman)
rm = riemannian_cluster(X)
# print(type(rm))
print(rm)
np.save("E:/博士成果/NIPS论文/SEED-IV/Structure/S{}.npy".format(k), rm)