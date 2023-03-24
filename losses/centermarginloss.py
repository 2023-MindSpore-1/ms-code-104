import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor, Parameter

# import losses.sigma.functional as F
from losses.sigma.utils import detect_large, split, split_2, split_KN_pos
import math
# class CenterMarginLoss(nn.Module):


class CenterLoss(nn.Cell):

    """
    paper: http://ydwen.github.io/papers/WenECCV16.pdf
    code:  https://github.com/pangyupo/mxnet_center_loss
    pytorch code: https://blog.csdn.net/sinat_37787331/article/details/80296964
    """
    def __init__(self, features_dim, num_class=200, lamda=0.01, scale=1., batch_size=64):
        """
        初始化
        :param features_dim: 特征维度 = c*h*w
        :param num_class: 类别数量
        :param lamda   centerloss的权重系数 [0,1]
        :param scale:  center 的梯度缩放因子
        :param batch_size:  批次大小
        """
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        # store the center of each class , should be ( num_class, features_dim)
        self.feature_centers = Parameter(np.randn([num_class, features_dim]))
        # print(self.feature_centers)
        # self.register_buffer('E', torch.eye(64))
        # self.lossfunc = CenterLossFunc.apply

    def forward(self, output_features, y_truth):
        """
        损失计算
        :param output_features: conv层输出的特征,  [b,c,h,w]
        :param y_truth:  标签值  [b,]
        :return:
        """
        batch_size = y_truth.size(0)
        # print(y_truth)
        output_features = output_features.view(batch_size, -1)
        # print(output_features.shape)
        # print(output_features.size(-1))
        assert output_features.size(-1) == self.feat_dim
        factor = self.scale / batch_size

        # print(self.feature_centers.shape)
        # tmp_y_truth = y_truth.unique()
        # diff_centers = self.feature_centers.index_select(0, tmp_y_truth.long())
        # print(diff_centers.shape)
        center_num = self.feature_centers.size(0)
        diff_centers_1 = self.feature_centers.view(center_num, 1, 200)
        diff_centers_2 = self.feature_centers.view(1, center_num, 200)
        loss2 = 410-self.lamda*(diff_centers_1 - diff_centers_2).pow(2).sum()/2/200
        # print(loss2.item())
        # loss2 = 0.5 * F.mse_loss(diff_centers.mm(diff_centers.t()), self.E[:center_num,:center_num])lf.feat_dim

        centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        # print(centers_batch.shape)
        diff = output_features - centers_batch
        loss = self.lamda * 0.5 * factor * (diff.pow(2).sum())
        # print(loss.item())
        #########
        return loss+loss2
    # """
    # paper: http://ydwen.github.io/papers/WenECCV16.pdf
    # code:  https://github.com/pangyupo/mxnet_center_loss
    # pytorch code: https://blog.csdn.net/sinat_37787331/article/details/80296964
    # """
    # def __init__(self,device, features_dim, num_class=200, lamda=0.01, scale=1., batch_size=64,  use_gpu=True):
    #     """
    #     初始化
    #     :param features_dim: 特征维度 = c*h*w
    #     :param num_class: 类别数量
    #     :param lamda   centerloss的权重系数 [0,1]
    #     :param scale:  center 的梯度缩放因子
    #     :param batch_size:  批次大小
    #     """
    #     super(CenterLoss, self).__init__()
    #     self.lamda = lamda
    #     self.num_class = num_class
    #     self.scale = scale
    #     self.batch_size = batch_size
    #     self.feat_dim = features_dim
    #     # store the center of each class , should be ( num_class, features_dim)
    #     self.feature_centers = nn.Parameter(torch.randn([num_class, features_dim]))
    #     # self.register_buffer('E', torch.eye(64))
    #     # self.lossfunc = CenterLossFunc.apply

    # def forward(self, output_features, y_truth, mode='t'):
    #     """
    #     损失计算
    #     :param output_features: conv层输出的特征,  [b,c,h,w]
    #     :param y_truth:  标签值  [b,]
    #     :return:
    #     """
    #     loss2 = 0
    #     loss = 0
    #     if mode == 't':
    #         output_features = output_features.view(batch_size, -1)
    #         batch_size = y_truth.size(0)
    #         centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
    #     # # print(centers_batch.shape)
    #         diff = output_features - centers_batch
    #         loss = self.lamda * 0.5 * factor * (diff.pow(2).sum())

    #     else:
    #         batch_size = len(y_truth)
    #     # print(y_truth)
    #     # print('output_features', output_features.shape)
        
    #     # print(output_features.shape)
    #     # print(output_features.size(-1))

    #     # assert output_features.size(-1) == self.feat_dim
    #     factor = self.scale / batch_size

    #     # print(self.feature_centers.shape)
    #     # tmp_y_truth = y_truth.unique()
    #     # diff_centers = self.feature_centers.index_select(0, tmp_y_truth.long())
    #     # print(diff_centers.shape)
    #     for i in range(batch_size):
    #         if len(y_truth[i]) == 1:
    #             pass
    #         else:
    #             # print(len(y_truth[i]))
    #             y_index = y_truth[i]
    #             center_num = self.feature_centers.size(0)
    #             diff_centers_1 = self.feature_centers.view(center_num, 1, 200)
    #             diff_centers_2 = self.feature_centers.view(1, center_num, 200)
    #             tempmatrix =(diff_centers_1 - diff_centers_2).pow(2)
    #             # print(tempmatrix[0].shape)
    #             # print('=============')
    #             inter_class_centert = tempmatrix[y_index, :,:]
    #             inter_class_center = inter_class_centert[:,y_index,:]
    #             # print(self.lamda*(inter_class_center).sum()/2)
    #             loss2 = loss2 +  410-self.lamda*(inter_class_center).sum()/2
    #             loss2 = loss2 + 410-self.lamda*(diff_centers_1 - diff_centers_2).pow(2).sum()/2
        
    #     # # print(loss2.item())
    #     # # loss2 = 0.5 * F.mse_loss(diff_centers.mm(diff_centers.t()), self.E[:center_num,:center_num])lf.feat_dim

    #     # centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
    #     # # print(centers_batch.shape)
    #     # diff = output_features - centers_batch
    #     # loss = self.lamda * 0.5 * factor * (diff.pow(2).sum())
    #     # print(loss.item())
    #     #########
    #     return loss+loss2

    # """Center loss.
    
    # Reference:
    # Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    # Args:
    #     num_classes (int): number of classes.
    #     feat_dim (int): feature dimension.
    # """
    # def __init__(self, device, num_classes=10, feat_dim=2, s=30.0, m=0.50, easy_margin=False, use_gpu=True):
    #     super(CenterMarginLoss, self).__init__()
    #     self.num_classes = num_classes
    #     self.feat_dim = feat_dim
    #     self.use_gpu = use_gpu
    #     self.s = s
    #     self.m = m
        
    #     self.tau = 0.1

    #     # self.easy_margin = easy_margin
    #     # self.cos_m = math.cos(m)
    #     # self.sin_m = math.sin(m)
    #     # self.th = math.cos(math.pi - m)
    #     # self.mm = math.sin(math.pi - m) * m
 
    #     if self.use_gpu:
    #         self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
    #     else:
    #         self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    #     self.centers = self.centers+self.tau
 
    # def forward(self, x, labels):
    #     """
    #     Args:
    #         x: feature matrix with shape (batch_size, feat_dim).
    #         labels: ground truth labels with shape (batch_size).
    #     """
    #     # x = x.float()
    #     # cosine = F.linear(F.normalize(x), F.normalize(self.weight)).float()
    #     # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    #     # phi = (cosine * self.cos_m - sine * self.sin_m).float()
    #     # if self.easy_margin:
    #     #     phi = torch.where(cosine > 0, phi, cosine)
    #     # else:
    #     #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    #     # # --------------------------- convert label to one-hot ---------------------------
    #     # # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    #     # one_hot = torch.zeros(cosine.size(), device='cuda')
    #     # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
    #     # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    #     # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    #     # output *= self.s
    #     batch_size = x.size(0)
    #     x = x.float()
    #     distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
    #               torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
    #     distmat.addmm_(1, -2, x, self.centers.t())
 
    #     classes = torch.arange(self.num_classes).long()
    #     if self.use_gpu: classes = classes.cuda()
    #     labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
    #     mask = labels.eq(classes.expand(batch_size, self.num_classes))
 
    #     dist = distmat * mask.float()
    #     loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
 
    #     return loss