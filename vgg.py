from typing import Type, Union, List, Optional
from mindspore import nn, train
from mindspore.common.initializer import Normal

from mindspore import load_checkpoint, load_param_into_net
import mindspore as ms

__all__ = ['VGG', 'vgg19', ]

cfg = {'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
       }


class VGG(nn.Cell):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, num_classes)
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        feat = self.features(x)
        x = self.maxpool(feat)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return feat, x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode='pad', padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(*layers)


def vgg19(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


if __name__ == '__main__':
    model = vgg19(pretrained=True)
    print(model)