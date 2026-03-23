"""
Kernel-level component: 5x5 Convolution with padding.
Applies a 5x5 convolution with padding=2 to preserve spatial dimensions.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone 5x5 convolution kernel."""
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        return self.conv(x)


batch_size = 10
in_channels = 16
out_channels = 48
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
    assert output.shape[2] == height and output.shape[3] == width, "Spatial dims should be preserved with padding=2"
    print("PASS")


if __name__ == "__main__":
    run_tests()
