"""
Step 2: Branch 3x3 -> Kernels
Refactored branch_3x3 fusion using kernel components only.

Parent: level_1_fusion/branch_3x3.py
Children:
  - conv2d_480x96_1x1_fp32.py (L0 kernel: 1x1 reduction)
  - conv2d_96x208_3x3_fp32.py (L0 kernel: 3x3 convolution)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))

from conv2d_480x96_1x1_fp32 import Model as Reduce
from conv2d_96x208_3x3_fp32 import Model as Conv3x3


class RefactoredModel(nn.Module):
    """
    Refactored branch_3x3: 1x1 reduce then 3x3 conv, using kernel children only.
    """
    def __init__(self, in_channels, reduce_3x3, out_3x3):
        super().__init__()
        self.reduce = Reduce(in_channels, reduce_3x3)
        self.conv = Conv3x3(reduce_3x3, out_3x3)

    def forward(self, x):
        x = self.reduce(x)
        x = self.conv(x)
        return x


in_channels = 480
reduce_3x3 = 96
out_3x3 = 208
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, reduce_3x3, out_3x3]
