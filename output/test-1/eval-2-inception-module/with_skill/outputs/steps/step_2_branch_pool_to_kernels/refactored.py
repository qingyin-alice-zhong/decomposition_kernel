"""
Step 2: Branch Pool -> Kernels
Refactored branch_pool fusion using kernel components only.

Parent: level_1_fusion/branch_pool.py
Children:
  - maxpool2d_3x3_fp32.py (L0 kernel: max pooling)
  - conv2d_480x64_1x1_fp32.py (L0 kernel: 1x1 projection)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))

from maxpool2d_3x3_fp32 import Model as MaxPool
from conv2d_480x64_1x1_fp32 import Model as Proj


class RefactoredModel(nn.Module):
    """
    Refactored branch_pool: max pool then 1x1 projection, using kernel children only.
    """
    def __init__(self, in_channels, pool_proj):
        super().__init__()
        self.pool = MaxPool()
        self.proj = Proj(in_channels, pool_proj)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        return x


in_channels = 480
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, pool_proj]
