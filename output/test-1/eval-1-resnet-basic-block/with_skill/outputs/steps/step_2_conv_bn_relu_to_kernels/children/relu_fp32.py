import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x)

def get_inputs():
    return [torch.randn(10, 64, 224, 224)]

def get_init_inputs():
    return []
