"""
Kernel-level component: 3x3 Convolution with padding.
Applies a 3x3 convolution with padding=1 to preserve spatial dimensions.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone 3x3 convolution kernel."""
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


batch_size = 10
in_channels = 96
out_channels = 208
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_channels, height, width), \
        f"Expected shape {(batch_size, out_channels, height, width)}, got {output.shape}"
    assert output.shape[2] == height and output.shape[3] == width, "Spatial dims should be preserved with padding=1"
    print("PASS")


if __name__ == "__main__":
    run_tests()
