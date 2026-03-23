"""
Component: SwiGLU Activation
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: MLPBlock

Operations: [SwiGLU - interleave split, clamp, sigmoid, multiply]

Input Shapes:
  - x: [32, 2, 5760] dtype=bfloat16

Output Shapes:
  - output: [32, 2, 2880] dtype=bfloat16
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, limit=7.0, alpha=1.702):
        super().__init__()
        self.limit = limit
        self.alpha = alpha

    def forward(self, x):
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(min=None, max=self.limit)
        x_linear = x_linear.clamp(min=-self.limit, max=self.limit)
        out_glu = x_glu * torch.sigmoid(self.alpha * x_glu)
        return out_glu * (x_linear + 1)


def get_inputs():
    return [torch.randn(32, 2, 5760, dtype=torch.bfloat16)]


def get_init_inputs():
    return [7.0, 1.702]


def get_expected_output_shape():
    return [(32, 2, 2880)]


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
