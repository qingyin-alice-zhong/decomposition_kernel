# Decomposition Analysis: ResNet BasicBlock

## Source
- File: `data/kernelbench/level3/8_ResNetBasicBlock.py`
- Model: ResNet BasicBlock with downsample

## Architecture Analysis

### Module Hierarchy
- Model (ResNet BasicBlock, Level 2 - Layer)
  - self.conv1: nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  - self.bn1: nn.BatchNorm2d(64)
  - self.relu: nn.ReLU(inplace=True)
  - self.conv2: nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
  - self.bn2: nn.BatchNorm2d(64)
  - self.downsample: nn.Sequential
    - 0: nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
    - 1: nn.BatchNorm2d(64)

### Data Flow (with shapes)

Input: [10, 3, 224, 224] (float32)

**Main path:**
```
x: [10, 3, 224, 224]
  -> conv1: [10, 3, 224, 224] -> [10, 64, 224, 224]
  -> bn1:   [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> relu:  [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> conv2: [10, 64, 224, 224] -> [10, 64, 224, 224]
  -> bn2:   [10, 64, 224, 224] -> [10, 64, 224, 224]
```

**Skip/downsample path:**
```
identity = x: [10, 3, 224, 224]
  -> downsample.0 (conv1x1): [10, 3, 224, 224] -> [10, 64, 224, 224]
  -> downsample.1 (bn):      [10, 64, 224, 224] -> [10, 64, 224, 224]
```

**Merge:**
```
out + identity: [10, 64, 224, 224] (residual add)
  -> relu: [10, 64, 224, 224] -> [10, 64, 224, 224]
```

Output: [10, 64, 224, 224] (float32)

### Parameters
- conv1.weight: (64, 3, 3, 3) = 1,728 params
- bn1.weight: (64,) = 64 params
- bn1.bias: (64,) = 64 params
- conv2.weight: (64, 64, 3, 3) = 36,864 params
- bn2.weight: (64,) = 64 params
- bn2.bias: (64,) = 64 params
- downsample.0.weight: (64, 3, 1, 1) = 192 params
- downsample.1.weight: (64,) = 64 params
- downsample.1.bias: (64,) = 64 params
- **Total: 39,168 params**

## Abstraction Level Classification

The original model is a **Level 2 (Layer)** -- a single building block that would typically be repeated in a full ResNet.

## Decomposition Strategy

### Round 1: Layer (L2) -> Fusions (L1) + Kernel (L0)

The BasicBlock contains three natural fusion groups plus one standalone kernel:

1. **conv_bn_relu** (L1 Fusion): conv1 -> bn1 -> relu
   - Straight-line computation, no data flow breaks
   - Natural fusion candidate for GPU kernel fusion

2. **conv_bn** (L1 Fusion): conv2 -> bn2
   - Straight-line computation
   - No activation because the residual add happens next

3. **downsample** (L1 Fusion): conv1x1 -> bn
   - Skip connection path
   - Straight-line computation

4. **relu** (L0 Kernel): Final ReLU after residual add
   - Standalone because it is separated from conv_bn by the residual add
   - Cannot be fused with conv_bn since the add must happen first

### Round 2: Each Fusion -> Kernels (L0)

Each fusion decomposes into individual operations:

- conv_bn_relu -> conv2d_3x64_3x3_fp32, batchnorm2d_64_fp32, relu_fp32
- conv_bn -> conv2d_64x64_3x3_fp32, batchnorm2d_64_fp32
- downsample -> conv2d_3x64_1x1_fp32, batchnorm2d_64_fp32

### Unique Kernel Files (5 total)

| Kernel File | Operation | Instances |
|---|---|---|
| conv2d_3x64_3x3_fp32.py | Conv2d(3, 64, 3x3, stride=1, padding=1) | 1 |
| conv2d_64x64_3x3_fp32.py | Conv2d(64, 64, 3x3, stride=1, padding=1) | 1 |
| conv2d_3x64_1x1_fp32.py | Conv2d(3, 64, 1x1, stride=1) | 1 |
| batchnorm2d_64_fp32.py | BatchNorm2d(64) | 3 |
| relu_fp32.py | ReLU(inplace=True) | 2 |
