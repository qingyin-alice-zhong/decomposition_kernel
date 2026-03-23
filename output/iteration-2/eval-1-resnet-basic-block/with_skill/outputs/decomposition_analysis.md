# Decomposition Analysis: ResNet BasicBlock

## Architecture Analysis

### Module Hierarchy
- Model (ResNet BasicBlock) [L2 Layer]
  - self.conv1: nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  - self.bn1: nn.BatchNorm2d(64)
  - self.relu: nn.ReLU(inplace=True) [shared for two uses]
  - self.conv2: nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
  - self.bn2: nn.BatchNorm2d(64)
  - self.downsample: nn.Sequential
    - nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
    - nn.BatchNorm2d(64)

### Data Flow (with shapes)
Input: [10, 3, 224, 224] (float32)
  identity = x                                     # [10, 3, 224, 224]
  -> conv1: [10, 3, 224, 224] -> [10, 64, 224, 224]
  -> bn1: [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> relu: [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> conv2: [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> bn2: [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> downsample(identity): [10, 3, 224, 224] -> [10, 64, 224, 224]
  -> out += identity: [10, 64, 224, 224]
  -> relu: [10, 64, 224, 224] -> [10, 64, 224, 224]
Output: [10, 64, 224, 224] (float32)

### Parameters
- conv1.weight: [64, 3, 3, 3]
- bn1.weight: [64], bn1.bias: [64], bn1.running_mean: [64], bn1.running_var: [64]
- conv2.weight: [64, 64, 3, 3]
- bn2.weight: [64], bn2.bias: [64], bn2.running_mean: [64], bn2.running_var: [64]
- downsample.0.weight: [64, 3, 1, 1]
- downsample.1.weight: [64], downsample.1.bias: [64], downsample.1.running_mean: [64], downsample.1.running_var: [64]

## Abstraction Level Classification
- The Model is a ResNet BasicBlock = Layer (L2)
- Decomposition: L2 -> L1 fusions + L0 kernels

### Decomposition Plan
Round 1: Layer (L2) -> Fusions (L1) + Kernels (L0)
  - conv_bn_relu: Conv2d(3,64,3x3) + BN(64) + ReLU [L1 fusion]
  - conv_bn: Conv2d(64,64,3x3) + BN(64) [L1 fusion]
  - downsample_conv_bn: Conv2d(3,64,1x1) + BN(64) [L1 fusion]
  - relu (after residual add): standalone ReLU [L0 kernel]

  Note: The residual add is data plumbing and stays in the refactored forward().

Round 2: Each Fusion (L1) -> Kernels (L0)
  - conv_bn_relu -> conv2d_3x64_3x3, batchnorm2d_64, relu_fp32
  - conv_bn -> conv2d_64x64_3x3, batchnorm2d_64 (reuse)
  - downsample_conv_bn -> conv2d_3x64_1x1, batchnorm2d_64 (reuse)

### Unique L0 Kernels
1. conv2d_3x64_3x3_fp32.py - Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
2. conv2d_64x64_3x3_fp32.py - Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
3. conv2d_3x64_1x1_fp32.py - Conv2d(3, 64, 1, stride=1, bias=False)
4. batchnorm2d_64_fp32.py - BatchNorm2d(64)
5. relu_fp32.py - ReLU(inplace=True)
