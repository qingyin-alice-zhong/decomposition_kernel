"""
Component: Downsample Conv+BN (3->64, 1x1)
Abstraction Level: fusion
Parent: resnet_basic_block
Children: [conv2d_3x64_1x1_fp32, batchnorm2d_64_fp32]

Operations: [Conv2d, BatchNorm2d]

Input Shapes:
  - x: (10, 3, 224, 224) dtype=float32

Output Shapes:
  - output: (10, 64, 224, 224) dtype=float32

Weight Shapes:
  - conv.weight: (64, 3, 1, 1)
  - bn.weight: (64,)
  - bn.bias: (64,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Downsample path: Conv2d(1x1) + BatchNorm2d.
    Extracted from: ResNet BasicBlock downsample sequential
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def get_inputs():
    return [torch.randn(10, 3, 224, 224)]

def get_init_inputs():
    return [3, 64, 1]

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
