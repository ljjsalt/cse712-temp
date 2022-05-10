from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Union, List, Dict, Any, Optional, cast


class PolyAct(nn.Module):
    def __init__(self, inplace=True) -> None:
        super().__init__()
        self.a = Parameter(torch.Tensor(1))
        self.b = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        nn.init.zeros_(self.a)
        nn.init.ones_(self.b)
        nn.init.zeros_(self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x**2 + self.b * x + self.c

act_fn = {
    'relu': nn.ReLU,
    'poly': PolyAct,
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.act1 = act_fn[act](inplace=True)
        self.act2 = act_fn[act](inplace=True)

    def forward(self, x):
        out = self.bn3(self.act1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.bn4(self.act2(out))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, act='relu'):
        super(ResNet, self).__init__()
        base = 16
        self.in_planes = base

        self.conv1 = nn.Conv2d(3, base, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, base, num_blocks[0], stride=1, act=act)
        self.layer2 = self._make_layer(block, base*2, num_blocks[1], stride=2, act=act)
        self.layer3 = self._make_layer(block, base*4, num_blocks[2], stride=2, act=act)
        self.layer4 = self._make_layer(block, base*8, num_blocks[2], stride=2, act=act)
        self.linear = nn.Linear(128*block.expansion, num_classes)
        self.act = act_fn[act](inplace=True)

    def _make_layer(self, block, planes, num_blocks, stride, act):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn2(self.act(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, 1)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def presnet10(num_classes, act):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, act)

def presnet14(num_classes, act):
    return ResNet(BasicBlock, [2, 2, 2], num_classes, act)
