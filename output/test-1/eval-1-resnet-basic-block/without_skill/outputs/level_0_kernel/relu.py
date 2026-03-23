"""
Level 0 (Kernel): ReLU activation
Atomic ReLU activation kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.relu(x)


# Test code
channels = 64
batch_size = 10


def get_inputs():
    return [torch.randn(batch_size, channels, 56, 56)]


def get_init_inputs():
    return []


def run_tests():
    model = Model().eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, channels, 56, 56), f"Shape mismatch: {output.shape}"
    assert (output >= 0).all(), "ReLU output should be non-negative"
    # Verify: positive inputs unchanged, negative inputs zeroed
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0]).view(1, 1, 2, 2)
    with torch.no_grad():
        y = model(x)
    assert torch.equal(y, torch.tensor([0.0, 0.0, 1.0, 2.0]).view(1, 1, 2, 2)), "ReLU behavior incorrect"
    print("PASS")


if __name__ == "__main__":
    run_tests()
