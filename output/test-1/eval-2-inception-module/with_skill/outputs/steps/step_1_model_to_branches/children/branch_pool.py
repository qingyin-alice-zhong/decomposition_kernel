"""
Component: Branch Pool (MaxPool2d + 1x1 conv projection)
Source: data/kernelbench/level3/6_GoogleNetInceptionModule.py
Abstraction Level: fusion
Parent: inception_module

Operations: [MaxPool2d(3, stride=1, padding=1), Conv2d(480, 64, 1)]

Input Shapes:
  - x: (10, 480, 224, 224) dtype=float32

Output Shapes:
  - output: (10, 64, 224, 224) dtype=float32

Weight Shapes:
  - proj.weight: (64, 480, 1, 1)
  - proj.bias: (64,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Pool branch of Inception module: 3x3 max pooling followed by 1x1 projection.
    Extracted from: inception_module
    """
    def __init__(self, in_channels, pool_proj):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        return x

# Test code
in_channels = 480
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, pool_proj]

def get_expected_output_shape():
    return [(batch_size, pool_proj, height, width)]

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
