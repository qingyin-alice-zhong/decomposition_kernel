"""
Step 2c: Downsample Conv+BN fusion (L1) -> Kernels (L0)

Decomposes downsample_conv_bn into:
  - conv2d_3x64_1x1_fp32 (L0)
  - batchnorm2d_64_fp32 (L0)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "children"))

from conv2d_3x64_1x1_fp32 import Model as Conv2d
from batchnorm2d_64_fp32 import Model as BatchNorm2d


class RefactoredModel(nn.Module):
    """
    Refactored downsample Conv+BN fusion using kernel modules.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 1, stride)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def get_inputs():
    return [torch.randn(10, 3, 224, 224)]

def get_init_inputs():
    return [3, 64, 1]


if __name__ == "__main__":
    model = RefactoredModel(*get_init_inputs())
    model.eval()
    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)
    print(f"Input shape: {inputs[0].shape}")
    print(f"Output shape: {output.shape}")
    print("PASS")
