# -*- coding: utf-8 -*-
import mindspore
from mindspore import nn,ops, Parameter, Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import GeneratorDataset

import random
import mindspore.numpy as np
from resnet import resnet50

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def Mask(nb_batch, channels):

    foo = [1] * 2 + [0] *  1
    bar = []
    for i in range(200):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,200*channels,1,1)
    bar = Tensor(bar)
    bar = bar.cuda()
    bar = Parameter(bar)
    return bar


def supervisor(x,targets,height,cnum):
        mask = Mask(x.size(0), cnum)
        branch = x
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        branch = ops.softmax(branch,2)
        branch = branch.reshape(branch.size(0),branch.size(1), x.size(2), x.size(2))
        branch = nn.MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        loss_2 = 1.0 - 1.0*ops.mean(np.sum(branch,2))/cnum # set margin = 3.0

        branch_1 = x * mask 

        branch_1 = nn.MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)
        branch_1 = nn.AvgPool2d(kernel_size=(height,height))(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)
        loss_1 = criterion(branch_1, targets)
        
        return [loss_1, loss_2]


def make_embedding_layer(in_features, sz_embedding, weight_init = None):
    embedding_layer = nn.Dense(in_features, sz_embedding)
    if weight_init != None: 
        weight_init(embedding_layer.weight)
    return embedding_layer


def bn_inception_weight_init(weight):
    import scipy.stats as stats
    stddev = 0.001
    X = stats.truncnorm(-2, 2, scale=stddev)
    values = Tensor(
        X.rvs(weight.data.numel())
    ).resize_(weight.size())
    weight.data.copy_(values)


class ResNet50(nn.Cell):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, pretrained=True,n_classes=200):
        super().__init__()

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.SequentialCell(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.conv1 = nn.Conv2d(2048, 3*self._n_classes, 3, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.embedding_layer = make_embedding_layer(
        2048,
        64,
        weight_init = bn_inception_weight_init
    )
        self.fc = nn.Dense(in_features=2048, out_features=self._n_classes)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(x)
        
        return  x# featx

