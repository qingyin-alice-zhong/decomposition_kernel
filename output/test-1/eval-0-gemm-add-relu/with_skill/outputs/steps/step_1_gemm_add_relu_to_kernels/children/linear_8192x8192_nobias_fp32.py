"""
Component: Linear (no bias)
Source: data/kernelbench/level2/76_Gemm_Add_ReLU.py
Abstraction Level: kernel
Parent: gemm_add_relu

Operations: [Linear (bias=False)]

Input Shapes:
  - x: (1024, 8192) dtype=float32

Output Shapes:
  - output: (1024, 8192) dtype=float32

Weight Shapes:
  - linear.weight: (8192, 8192)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Linear projection without bias.

    Extracted from: gemm_add_relu fusion
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]

def get_expected_output_shape():
    return [(batch_size, out_features)]
