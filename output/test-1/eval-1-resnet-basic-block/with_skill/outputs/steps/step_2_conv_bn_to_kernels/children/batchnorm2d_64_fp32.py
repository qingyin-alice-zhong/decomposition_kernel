import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)

def get_inputs():
    return [torch.randn(10, 64, 224, 224)]

def get_init_inputs():
    return [64]
