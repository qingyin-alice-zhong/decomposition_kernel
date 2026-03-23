"""
Level 0 (Kernel): 1x1 Conv2d
Atomic 1x1 convolution kernel used in the downsample/skip connection path.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False)

    def forward(self, x):
        return self.conv(x)


# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10


def get_inputs():
    return [torch.rand(batch_size, in_channels, 56, 56)]


def get_init_inputs():
    return [in_channels, out_channels, stride]


def run_tests():
    model = Model(*get_init_inputs()).eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, out_channels, 56, 56), f"Shape mismatch: {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
