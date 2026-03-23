"""
Refactored ResNet BasicBlock: Layer -> Fusions + Kernel

Replaces inline conv/bn/relu with child module calls.
Children:
  - conv_bn_relu: Conv2d + BatchNorm2d + ReLU (first path)
  - conv_bn: Conv2d + BatchNorm2d (second path)
  - downsample: Conv2d(1x1) + BatchNorm2d (skip connection)
  - relu_fp32: ReLU (standalone, after residual add)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))
from conv_bn_relu import Model as ConvBnRelu
from conv_bn import Model as ConvBn
from downsample import Model as Downsample
from relu_fp32 import Model as ReLU

class RefactoredModel(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_bn_relu = ConvBnRelu(in_channels, out_channels, 3, stride, 1)
        self.conv_bn = ConvBn(out_channels, out_channels, 3, 1, 1)
        self.downsample_block = Downsample(in_channels, out_channels, stride)
        self.final_relu = ReLU()

    def forward(self, x):
        identity = x

        out = self.conv_bn_relu(x)
        out = self.conv_bn(out)

        identity = self.downsample_block(identity)

        out = out + identity
        out = self.final_relu(out)

        return out

in_channels = 3
out_channels = 64
stride = 1
batch_size = 10

def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]
