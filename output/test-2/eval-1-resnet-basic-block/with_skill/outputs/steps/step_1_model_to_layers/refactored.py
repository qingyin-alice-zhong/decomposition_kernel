"""
Step 1: ResNet BasicBlock (L2) -> Fusions (L1) + Kernel (L0)

Decomposes the BasicBlock into:
  - conv_bn_relu (L1): conv1 + bn1 + relu
  - conv_bn (L1): conv2 + bn2
  - downsample_conv_bn (L1): downsample conv + bn
  - relu (L0): standalone relu after residual add
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))

from conv_bn_relu_3x64_3x3_fp32 import Model as ConvBnRelu
from conv_bn_64x64_3x3_fp32 import Model as ConvBn
from downsample_conv_bn_3x64_1x1_fp32 import Model as DownsampleConvBn
from relu_fp32 import Model as Relu


class RefactoredModel(nn.Module):
    """
    Refactored ResNet BasicBlock using child fusion/kernel modules.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_bn_relu = ConvBnRelu(in_channels, out_channels, 3, stride, 1)
        self.conv_bn = ConvBn(out_channels, out_channels, 3, 1, 1)
        self.downsample = DownsampleConvBn(in_channels, out_channels * self.expansion, stride)
        self.final_relu = Relu()

    def forward(self, x):
        identity = x

        out = self.conv_bn_relu(x)
        out = self.conv_bn(out)

        identity = self.downsample(identity)

        out += identity
        out = self.final_relu(out)

        return out


in_channels = 3
out_channels = 64
stride = 1
batch_size = 10

def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]


if __name__ == "__main__":
    model = RefactoredModel(*get_init_inputs())
    model.eval()
    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)
    print(f"Input shape: {inputs[0].shape}")
    print(f"Output shape: {output.shape}")
    print("PASS")
