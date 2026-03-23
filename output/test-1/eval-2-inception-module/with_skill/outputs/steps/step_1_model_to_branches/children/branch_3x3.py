"""
Component: Branch 3x3 (1x1 reduce + 3x3 conv)
Source: data/kernelbench/level3/6_GoogleNetInceptionModule.py
Abstraction Level: fusion
Parent: inception_module

Operations: [Conv2d(480, 96, 1), Conv2d(96, 208, 3, padding=1)]

Input Shapes:
  - x: (10, 480, 224, 224) dtype=float32

Output Shapes:
  - output: (10, 208, 224, 224) dtype=float32

Weight Shapes:
  - reduce.weight: (96, 480, 1, 1)
  - reduce.bias: (96,)
  - conv.weight: (208, 96, 3, 3)
  - conv.bias: (208,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3x3 branch of Inception module: 1x1 reduction followed by 3x3 convolution.
    Extracted from: inception_module
    """
    def __init__(self, in_channels, reduce_3x3, out_3x3):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.conv = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.reduce(x)
        x = self.conv(x)
        return x

# Test code
in_channels = 480
reduce_3x3 = 96
out_3x3 = 208
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, reduce_3x3, out_3x3]

def get_expected_output_shape():
    return [(batch_size, out_3x3, height, width)]

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
            actual_shapes = [output.shape]
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
