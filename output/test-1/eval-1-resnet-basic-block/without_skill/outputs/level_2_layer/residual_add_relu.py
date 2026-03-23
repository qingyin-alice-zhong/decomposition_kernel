"""
Level 2 (Layer): Residual Add + ReLU
Adds the skip connection (identity or downsampled) to the main path output, then applies ReLU.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, main_path, identity):
        out = main_path + identity
        out = self.relu(out)
        return out


# Test code
channels = 64
batch_size = 10


def get_inputs():
    return [torch.randn(batch_size, channels, 224, 224),
            torch.randn(batch_size, channels, 224, 224)]


def get_init_inputs():
    return []


def run_tests():
    model = Model().eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, channels, 224, 224), f"Shape mismatch: {output.shape}"
    assert (output >= 0).all(), "Output after ReLU should be non-negative"
    print("PASS")


if __name__ == "__main__":
    run_tests()
