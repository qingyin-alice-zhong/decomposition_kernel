"""
Component: Conv2d 96->208 kernel_size=3 padding=1 (3x3 branch convolution)
Source: data/kernelbench/level3/6_GoogleNetInceptionModule.py
Abstraction Level: kernel
Parent: branch_3x3

Operations: [Conv2d]

Input Shapes:
  - x: (10, 96, 224, 224) dtype=float32

Output Shapes:
  - output: (10, 208, 224, 224) dtype=float32

Weight Shapes:
  - conv.weight: (208, 96, 3, 3)
  - conv.bias: (208,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3x3 convolution: 96 -> 208 channels with padding=1.
    Extracted from: branch_3x3
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

in_channels = 96
out_channels = 208
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]

def get_expected_output_shape():
    return [(batch_size, out_channels, height, width)]

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
