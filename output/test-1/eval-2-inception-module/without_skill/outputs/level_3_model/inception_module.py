"""
Model-level component: GoogleNet Inception Module.
The full Inception module with 4 parallel branches concatenated along channels:
  1. 1x1 convolution
  2. 1x1 reduction + 3x3 convolution
  3. 1x1 reduction + 5x5 convolution
  4. 3x3 max pooling + 1x1 projection
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(Model, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    total_out = out_1x1 + out_3x3 + out_5x5 + pool_proj
    assert output.shape == (batch_size, total_out, height, width), \
        f"Expected shape {(batch_size, total_out, height, width)}, got {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
