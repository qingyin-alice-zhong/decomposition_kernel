"""
Component: BatchNorm2d (64 channels)
Source: data/kernelbench/level3/8_ResNetBasicBlock.py
Abstraction Level: kernel
Parent: conv_bn_relu, conv_bn, downsample

Operations: [BatchNorm2d]

Input Shapes:
  - x: (10, 64, 224, 224) dtype=float32

Output Shapes:
  - output: (10, 64, 224, 224) dtype=float32

Weight Shapes:
  - bn.weight: (64,)
  - bn.bias: (64,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Single BatchNorm2d operation with 64 channels.

    Extracted from: resnet_basic_block (bn1, bn2, downsample.1)
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)

def get_inputs():
    return [torch.randn(10, 64, 224, 224)]

def get_init_inputs():
    return [64]

def get_expected_output_shape():
    return [(10, 64, 224, 224)]

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
