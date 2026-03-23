"""
Fusion-level component: MaxPool + 1x1 Projection.
Fuses a 3x3 max pooling (stride=1, padding=1) followed by a 1x1 projection convolution.
This is the core computation pattern of the pooling branch in an Inception module.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Fused max pooling followed by 1x1 projection convolution."""
    def __init__(self, in_channels, proj_channels):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        return x


batch_size = 10
in_channels = 480
proj_channels = 64
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, proj_channels]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, proj_channels, height, width), \
        f"Expected shape {(batch_size, proj_channels, height, width)}, got {output.shape}"
    # Verify pooling preserves spatial dims
    x = inputs[0]
    pooled = model.pool(x)
    assert pooled.shape == (batch_size, in_channels, height, width), \
        f"Pooling should preserve spatial dims"
    print("PASS")


if __name__ == "__main__":
    run_tests()
