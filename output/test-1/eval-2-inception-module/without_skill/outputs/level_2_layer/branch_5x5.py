"""
Layer-level component: 5x5 Convolution Branch.
A two-stage branch: 1x1 reduction to reduce channels, then 5x5 convolution
for spatial feature extraction. Captures large-scale spatial patterns.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Inception 5x5 branch as a standalone layer."""
    def __init__(self, in_channels, reduce_5x5, out_5x5):
        super(Model, self).__init__()
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.branch5x5(x)


batch_size = 10
in_channels = 480
reduce_5x5 = 16
out_5x5 = 48
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, reduce_5x5, out_5x5]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_5x5, height, width), \
        f"Expected shape {(batch_size, out_5x5, height, width)}, got {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
