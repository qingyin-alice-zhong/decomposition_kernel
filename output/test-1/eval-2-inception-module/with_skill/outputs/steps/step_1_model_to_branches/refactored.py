"""
Step 1: Model -> Branches
Refactored inception module using child components only.

Parent: level_3_model/inception_module.py
Children:
  - conv2d_480x192_1x1_fp32.py (branch1x1, L0 kernel)
  - branch_3x3.py (L1 fusion)
  - branch_5x5.py (L1 fusion)
  - branch_pool.py (L1 fusion)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))

from conv2d_480x192_1x1_fp32 import Model as Branch1x1
from branch_3x3 import Model as Branch3x3
from branch_5x5 import Model as Branch5x5
from branch_pool import Model as BranchPool


class RefactoredModel(nn.Module):
    """
    Refactored GoogleNet Inception Module using child components.
    forward() only calls child modules + data plumbing (torch.cat).
    """
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()
        self.branch1x1 = Branch1x1(in_channels, out_1x1)
        self.branch3x3 = Branch3x3(in_channels, reduce_3x3, out_3x3)
        self.branch5x5 = Branch5x5(in_channels, reduce_5x5, out_5x5)
        self.branch_pool = BranchPool(in_channels, pool_proj)

    def forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], 1)


# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
