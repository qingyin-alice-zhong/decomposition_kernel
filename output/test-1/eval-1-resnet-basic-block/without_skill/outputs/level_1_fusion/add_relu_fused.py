"""
Level 1 (Fusion): Fused Add + ReLU
Element-wise addition followed by ReLU activation, a common fusion target in deep learning compilers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Fused add + relu: max(x + y, 0)
        return F.relu(x + y, inplace=False)


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
    assert (output >= 0).all(), "Output should be non-negative after ReLU"
    # Verify equivalence to separate add + relu
    with torch.no_grad():
        expected = F.relu(inputs[0] + inputs[1])
    assert torch.allclose(output, expected), "Fused add+relu should match separate operations"
    print("PASS")


if __name__ == "__main__":
    run_tests()
