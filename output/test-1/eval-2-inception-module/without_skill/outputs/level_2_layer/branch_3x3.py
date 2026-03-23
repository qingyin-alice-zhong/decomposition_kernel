"""
Layer-level component: 3x3 Convolution Branch.
A two-stage branch: 1x1 reduction to reduce channels, then 3x3 convolution
for spatial feature extraction. Captures medium-scale spatial patterns.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Inception 3x3 branch as a standalone layer."""
    def __init__(self, in_channels, reduce_3x3, out_3x3):
        super(Model, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.branch3x3(x)


batch_size = 10
in_channels = 480
reduce_3x3 = 96
out_3x3 = 208
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, reduce_3x3, out_3x3]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_3x3, height, width), \
        f"Expected shape {(batch_size, out_3x3, height, width)}, got {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
