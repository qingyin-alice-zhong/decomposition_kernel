"""
Refactored: Gemm_Add_ReLU -> Kernels
Step 1: Decompose the fusion into individual kernel operations.

Original: Linear(8192, 8192, bias=False) + BiasAdd(8192) + ReLU
Children: linear_8192x8192_nobias_fp32, bias_add_8192_fp32, relu_fp32
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add children directory to path
sys.path.insert(0, str(Path(__file__).parent / "children"))

from linear_8192x8192_nobias_fp32 import Model as LinearNoBias
from bias_add_8192_fp32 import Model as BiasAdd
from relu_fp32 import Model as ReLU

class RefactoredModel(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = LinearNoBias(in_features, out_features)
        self.bias_add = BiasAdd(bias_shape)
        self.relu = ReLU()

    def forward(self, x):
        x = self.gemm(x)
        x = self.bias_add(x)
        x = self.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape]
