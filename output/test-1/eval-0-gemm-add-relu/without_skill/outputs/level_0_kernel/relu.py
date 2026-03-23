"""
Kernel-level component: ReLU activation.
Applies element-wise ReLU: max(0, x).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone ReLU activation operation."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.relu(x)


batch_size = 1024
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, out_features) - 0.5]  # mix of positive and negative


def get_init_inputs():
    return []


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"
    # Verify no negative values in output
    assert (output >= 0).all(), "ReLU output contains negative values"
    # Verify positive values are unchanged
    x = inputs[0]
    pos_mask = x > 0
    assert torch.allclose(output[pos_mask], x[pos_mask]), "ReLU changed positive values"
    print("PASS")


if __name__ == "__main__":
    run_tests()
