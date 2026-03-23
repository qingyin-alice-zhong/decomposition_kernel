"""
Component: Downsample Fusion (Conv1x1 + BN)
Abstraction Level: fusion
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def get_inputs():
    return [torch.randn(10, 3, 224, 224)]

def get_init_inputs():
    return [3, 64, 1]
