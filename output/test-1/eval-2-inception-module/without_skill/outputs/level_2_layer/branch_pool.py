"""
Layer-level component: Pooling Branch.
Max pooling followed by 1x1 projection. Provides pooled features at
reduced channel dimensionality as a complementary signal to the conv branches.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Inception pooling branch as a standalone layer."""
    def __init__(self, in_channels, pool_proj):
        super(Model, self).__init__()
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        return self.branch_pool(x)


batch_size = 10
in_channels = 480
pool_proj = 64
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, pool_proj]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, pool_proj, height, width), \
        f"Expected shape {(batch_size, pool_proj, height, width)}, got {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
