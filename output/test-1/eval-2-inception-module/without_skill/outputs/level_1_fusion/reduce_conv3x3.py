"""
Fusion-level component: 1x1 Reduction + 3x3 Convolution.
Fuses a 1x1 channel reduction convolution followed by a 3x3 convolution.
This is the core computation pattern of the 3x3 branch in an Inception module.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Fused 1x1 reduction followed by 3x3 convolution."""
    def __init__(self, in_channels, reduce_channels, out_channels):
        super(Model, self).__init__()
        self.reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.conv = nn.Conv2d(reduce_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.reduce(x)
        x = self.conv(x)
        return x


batch_size = 10
in_channels = 480
reduce_channels = 96
out_channels = 208
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, reduce_channels, out_channels]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_channels, height, width), \
        f"Expected shape {(batch_size, out_channels, height, width)}, got {output.shape}"
    # Verify intermediate reduction shape
    x = inputs[0]
    reduced = model.reduce(x)
    assert reduced.shape == (batch_size, reduce_channels, height, width), \
        f"Reduction output shape mismatch"
    print("PASS")


if __name__ == "__main__":
    run_tests()
