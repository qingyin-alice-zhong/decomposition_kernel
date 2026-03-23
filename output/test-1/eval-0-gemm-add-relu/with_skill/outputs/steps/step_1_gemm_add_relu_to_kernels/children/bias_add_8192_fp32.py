"""
Component: Bias Add
Source: data/kernelbench/level2/76_Gemm_Add_ReLU.py
Abstraction Level: kernel
Parent: gemm_add_relu

Operations: [BiasAdd]

Input Shapes:
  - x: (1024, 8192) dtype=float32

Output Shapes:
  - output: (1024, 8192) dtype=float32

Weight Shapes:
  - bias: (8192,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adds a learnable bias vector to the input tensor.

    Extracted from: gemm_add_relu fusion
    """
    def __init__(self, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return x + self.bias

batch_size = 1024
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [bias_shape]

def get_expected_output_shape():
    return [(batch_size, out_features)]
