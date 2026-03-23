"""
Level 2 (Layer): Conv2d -> BatchNorm2d
Second layer in the BasicBlock main path (no ReLU, applied after residual add).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


# Test code
in_channels = 64
out_channels = 64
stride = 1
batch_size = 10


def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]


def get_init_inputs():
    return [in_channels, out_channels, 3, stride, 1]


def run_tests():
    model = Model(*get_init_inputs()).eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, out_channels, 224, 224), f"Shape mismatch: {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
