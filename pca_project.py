import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np


class PCAProjectNet(nn.Cell):
    def __init__(self):
        super(PCAProjectNet, self).__init__()

    def construct(self, features):     # features: NCWH
        k = features.size(0) * features.size(2) * features.size(3)
        x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        features = features - x_mean

        reshaped_features = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)

        cov = ops.matmul(reshaped_features, reshaped_features.t()) / k
        eigval, eigvec = ops.eig(cov, eigenvectors=True)#特征值和特征向量

        first_compo = eigvec[:, 0]#取特征向量第一维

        projected_map = ops.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
            .view(features.size(0), features.size(2), features.size(3))

        maxv = projected_map.max()
        minv = projected_map.min()

        projected_map *= (maxv + minv) / ops.abs(maxv + minv)
        
        return projected_map

