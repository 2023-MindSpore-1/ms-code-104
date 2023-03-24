import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor, Parameter

import losses.sigma.functional as F
from losses.sigma.utils import detect_large, split, split_2, split_KN_pos


class CenterLoss(nn.Cell):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, device, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
 
        if self.use_gpu:
            self.centers = Parameter(np.randn(self.num_classes, self.feat_dim))
        else:
            self.centers = Parameter(np.randn(self.num_classes, self.feat_dim))
 
    def construct(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        x = x.float()
        distmat = ops.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  ops.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
 
        classes = ops.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
 
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
 
        return loss