"""
Model-level component: The complete Gemm_Add_ReLU model.
This is the top-level model that matches the original 76_Gemm_Add_ReLU.py exactly.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = x + self.bias
        x = torch.relu(x)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"
    assert (output >= 0).all(), "Output contains negative values"
    print("PASS")


if __name__ == "__main__":
    run_tests()
