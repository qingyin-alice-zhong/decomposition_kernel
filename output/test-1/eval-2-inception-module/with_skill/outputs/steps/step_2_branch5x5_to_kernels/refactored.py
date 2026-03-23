"""
Step 2: Branch 5x5 -> Kernels
Refactored branch_5x5 fusion using kernel components only.

Parent: level_1_fusion/branch_5x5.py
Children:
  - conv2d_480x16_1x1_fp32.py (L0 kernel: 1x1 reduction)
  - conv2d_16x48_5x5_fp32.py (L0 kernel: 5x5 convolution)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))

from conv2d_480x16_1x1_fp32 import Model as Reduce
from conv2d_16x48_5x5_fp32 import Model as Conv5x5


class RefactoredModel(nn.Module):
    """
    Refactored branch_5x5: 1x1 reduce then 5x5 conv, using kernel children only.
    """
    def __init__(self, in_channels, reduce_5x5, out_5x5):
        super().__init__()
        self.reduce = Reduce(in_channels, reduce_5x5)
        self.conv = Conv5x5(reduce_5x5, out_5x5)

    def forward(self, x):
        x = self.reduce(x)
        x = self.conv(x)
        return x


in_channels = 480
reduce_5x5 = 16
out_5x5 = 48
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, reduce_5x5, out_5x5]
