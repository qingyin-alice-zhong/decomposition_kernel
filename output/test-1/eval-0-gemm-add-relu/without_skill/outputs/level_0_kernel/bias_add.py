"""
Kernel-level component: Bias addition.
Adds a learnable bias vector to the input tensor.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone bias addition operation."""
    def __init__(self, bias_shape):
        super(Model, self).__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return x + self.bias


batch_size = 1024
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, out_features)]


def get_init_inputs():
    return [bias_shape]


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"
    # Verify bias is actually added (output != input)
    x = inputs[0]
    diff = output - x
    assert torch.allclose(diff, model.bias.unsqueeze(0).expand_as(diff), atol=1e-6), "Bias not correctly added"
    print("PASS")


if __name__ == "__main__":
    run_tests()
