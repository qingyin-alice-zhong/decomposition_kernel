"""
Level 0 (Kernel): BatchNorm2d
Atomic batch normalization kernel.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


# Test code
num_features = 64
batch_size = 10


def get_inputs():
    return [torch.randn(batch_size, num_features, 56, 56)]


def get_init_inputs():
    return [num_features]


def run_tests():
    model = Model(*get_init_inputs())
    # Run a forward pass in train mode to populate running stats
    inputs = get_inputs()
    _ = model(*inputs)
    model.eval()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, num_features, 56, 56), f"Shape mismatch: {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
