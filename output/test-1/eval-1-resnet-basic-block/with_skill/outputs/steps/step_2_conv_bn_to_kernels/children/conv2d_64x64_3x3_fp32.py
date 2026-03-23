import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)

def get_inputs():
    return [torch.randn(10, 64, 224, 224)]

def get_init_inputs():
    return [64, 64, 3, 1, 1]
