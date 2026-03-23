"""
Component: RMSNorm
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: GPT-OSS Transformer / AttentionBlock / MLPBlock

Operations: [RMSNorm - pow, mean, rsqrt, multiply]

Input Shapes:
  - x: [32, 2880] dtype=bfloat16

Output Shapes:
  - output: [32, 2880] dtype=bfloat16

Weight Shapes:
  - scale: [2880]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def get_inputs():
    return [torch.randn(32, 2880, dtype=torch.bfloat16)]


def get_init_inputs():
    return [2880]


def get_expected_output_shape():
    return [(32, 2880)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
