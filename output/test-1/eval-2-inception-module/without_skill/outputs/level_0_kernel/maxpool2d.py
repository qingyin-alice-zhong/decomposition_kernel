"""
Kernel-level component: 3x3 Max Pooling.
Applies 3x3 max pooling with stride=1 and padding=1 to preserve spatial dimensions.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone 3x3 max pooling kernel."""
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.pool(x)


batch_size = 10
in_channels = 480
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return []


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, in_channels, height, width), \
        f"Expected shape {(batch_size, in_channels, height, width)}, got {output.shape}"
    # Verify max pooling produces values >= input (each output is max of neighborhood)
    x = inputs[0]
    assert (output >= x).all(), "Max pool output should be >= center element"
    print("PASS")


if __name__ == "__main__":
    run_tests()
