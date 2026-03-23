"""
Component: QKV Projection Linear
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: AttentionBlock

Operations: [nn.Linear]

Input Shapes:
  - x: [32, 2880] dtype=bfloat16

Output Shapes:
  - output: [32, 5120] dtype=bfloat16

Weight Shapes:
  - linear.weight: [5120, 2880]
  - linear.bias: [5120]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(32, 2880, dtype=torch.bfloat16)]


def get_init_inputs():
    return [2880, 5120]


def get_expected_output_shape():
    return [(32, 5120)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
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
