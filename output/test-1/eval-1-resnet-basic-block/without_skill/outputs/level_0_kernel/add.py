"""
Level 0 (Kernel): Element-wise Add
Atomic element-wise addition kernel used for residual connections.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return x + y


# Test code
channels = 64
batch_size = 10


def get_inputs():
    return [torch.randn(batch_size, channels, 56, 56),
            torch.randn(batch_size, channels, 56, 56)]


def get_init_inputs():
    return []


def run_tests():
    model = Model().eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, channels, 56, 56), f"Shape mismatch: {output.shape}"
    # Verify correctness
    assert torch.allclose(output, inputs[0] + inputs[1]), "Add operation incorrect"
    print("PASS")


if __name__ == "__main__":
    run_tests()
