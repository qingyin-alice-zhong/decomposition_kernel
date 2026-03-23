"""
Layer-level component: 1x1 Convolution Branch.
The simplest branch of the Inception module -- a single 1x1 convolution
that captures cross-channel correlations without spatial context.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Inception 1x1 branch as a standalone layer."""
    def __init__(self, in_channels, out_1x1):
        super(Model, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

    def forward(self, x):
        return self.branch1x1(x)


batch_size = 10
in_channels = 480
out_1x1 = 192
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_1x1]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_1x1, height, width), \
        f"Expected shape {(batch_size, out_1x1, height, width)}, got {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
