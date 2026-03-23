"""
Component: Bias Add
Source: data/kernelbench/level2/76_Gemm_Add_ReLU.py
Abstraction Level: kernel
Parent: gemm_add_relu

Operations: [BiasAdd]

Input Shapes:
  - x: (1024, 8192) dtype=float32

Output Shapes:
  - output: (1024, 8192) dtype=float32

Weight Shapes:
  - bias: (8192,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adds a learnable bias vector to the input tensor.

    Extracted from: gemm_add_relu fusion
    """
    def __init__(self, bias_shape):
        super().__init__()
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

def get_expected_output_shape():
    return [(batch_size, out_features)]

def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if isinstance(output, torch.Tensor) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"
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
