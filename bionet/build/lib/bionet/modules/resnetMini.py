'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from bionet.biomodule import BioModule as bm


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        base_channels = int((num_channels * 4) + 2)
        self.in_planes = int(base_channels / 10 * num_classes)

        self.conv1 = nn.Conv2d(num_channels, int(base_channels / 10 * num_classes), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(base_channels / 10 * num_classes))
        self.layer1 = self._make_layer(block, int(base_channels / 10 * num_classes), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(base_channels / 10 * num_classes * 2), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(base_channels / 10 * num_classes * 4), num_blocks[2], stride=2)
        
        self.linear = bm.mark_skip_accum(nn.Linear(int(base_channels / 10 * num_classes * 4), num_classes))
        

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

def resnet20(num_channels=3, num_classes=10):
    return ResNet,{"block":BasicBlock, "num_blocks":[3, 3, 3],"num_channels":num_channels, "num_classes":num_classes}


def resnet32(num_channels=3, num_classes=10):
    return ResNet,{"block":BasicBlock, "num_blocks":[5, 5, 5],"num_channels":num_channels, "num_classes":num_classes}


def resnet44(num_channels=3, num_classes=10):
    return ResNet,{"block":BasicBlock, "num_blocks":[7, 7, 7],"num_channels":num_channels, "num_classes":num_classes}


def resnet56(num_channels=3, num_classes=10):
    return ResNet,{"block":BasicBlock, "num_blocks":[9, 9, 9],"num_channels":num_channels, "num_classes":num_classes}


def resnet110(num_channels=3, num_classes=10):
    return ResNet,{"block":BasicBlock, "num_blocks":[18, 18, 18],"num_channels":num_channels, "num_classes":num_classes}


def resnet1202(num_channels=3, num_classes=10):
    return ResNet,{"block":BasicBlock, "num_blocks":[200, 200, 200],"num_channels":num_channels, "num_classes":num_classes}


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()