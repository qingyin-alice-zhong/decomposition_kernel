"""
Component: ResNet BasicBlock
Source: data/kernelbench/level3/8_ResNetBasicBlock.py
Abstraction Level: layer
Parent: root

Operations: [Conv2d, BatchNorm2d, ReLU, residual add, downsample]

Input Shapes:
  - x: (10, 3, 224, 224) dtype=float32

Output Shapes:
  - output: (10, 64, 224, 224) dtype=float32

Weight Shapes:
  - conv1.weight: (64, 3, 3, 3)
  - bn1.weight: (64,)
  - bn1.bias: (64,)
  - conv2.weight: (64, 64, 3, 3)
  - bn2.weight: (64,)
  - bn2.bias: (64,)
  - downsample.0.weight: (64, 3, 1, 1)
  - downsample.1.weight: (64,)
  - downsample.1.bias: (64,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    ResNet BasicBlock with two conv-bn paths and a residual/skip connection
    with downsample.

    Extracted from: data/kernelbench/level3/8_ResNetBasicBlock.py
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

in_channels = 3
out_channels = 64
stride = 1
batch_size = 10

def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]

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
