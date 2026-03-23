"""
Component: ReLU
Source: data/kernelbench/level2/76_Gemm_Add_ReLU.py
Abstraction Level: kernel
Parent: gemm_add_relu

Operations: [ReLU]

Input Shapes:
  - x: (1024, 8192) dtype=float32

Output Shapes:
  - output: (1024, 8192) dtype=float32

Weight Shapes:
  (none - elementwise activation)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReLU activation function.

    Extracted from: gemm_add_relu fusion
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

batch_size = 1024
features = 8192

def get_inputs():
    return [torch.randn(batch_size, features)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(batch_size, features)]
