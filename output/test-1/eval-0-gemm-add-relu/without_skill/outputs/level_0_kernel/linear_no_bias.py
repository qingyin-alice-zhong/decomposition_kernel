"""
Kernel-level component: Linear transformation (matrix multiply) without bias.
Corresponds to nn.Linear(in_features, out_features, bias=False).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone linear (GEMM) operation without bias."""
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.gemm(x)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"
    print("PASS")


if __name__ == "__main__":
    run_tests()
