"""
Level 2 (Layer): Downsample path (1x1 Conv2d -> BatchNorm2d)
Used to match dimensions for the residual/skip connection.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.downsample(x)


# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10


def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]


def get_init_inputs():
    return [in_channels, out_channels, stride]


def run_tests():
    model = Model(*get_init_inputs()).eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, out_channels, 224, 224), f"Shape mismatch: {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
