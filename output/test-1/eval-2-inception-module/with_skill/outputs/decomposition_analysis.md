# Decomposition Analysis: GoogleNet Inception Module

## Source
- File: `data/kernelbench/level3/6_GoogleNetInceptionModule.py`
- Model: GoogleNet Inception Module with 4 parallel branches

## Architecture Analysis

### Module Hierarchy
- Model (L3)
  - self.branch1x1: nn.Conv2d(480, 192, kernel_size=1) -- single op, L0 kernel
  - self.branch3x3: nn.Sequential -- L1 fusion
    - [0]: nn.Conv2d(480, 96, kernel_size=1) -- 1x1 reduction
    - [1]: nn.Conv2d(96, 208, kernel_size=3, padding=1) -- 3x3 conv
  - self.branch5x5: nn.Sequential -- L1 fusion
    - [0]: nn.Conv2d(480, 16, kernel_size=1) -- 1x1 reduction
    - [1]: nn.Conv2d(16, 48, kernel_size=5, padding=2) -- 5x5 conv
  - self.branch_pool: nn.Sequential -- L1 fusion
    - [0]: nn.MaxPool2d(kernel_size=3, stride=1, padding=1) -- max pooling
    - [1]: nn.Conv2d(480, 64, kernel_size=1) -- 1x1 projection

### Data Flow (with shapes)
```
Input: [10, 480, 224, 224] (float32)
  |
  +---> branch1x1: Conv2d(480, 192, 1) --> [10, 192, 224, 224]
  |
  +---> branch3x3:
  |       Conv2d(480, 96, 1) --> [10, 96, 224, 224]
  |       Conv2d(96, 208, 3, p=1) --> [10, 208, 224, 224]
  |
  +---> branch5x5:
  |       Conv2d(480, 16, 1) --> [10, 16, 224, 224]
  |       Conv2d(16, 48, 5, p=2) --> [10, 48, 224, 224]
  |
  +---> branch_pool:
          MaxPool2d(3, s=1, p=1) --> [10, 480, 224, 224]
          Conv2d(480, 64, 1) --> [10, 64, 224, 224]
  |
  cat(dim=1) --> [10, 512, 224, 224]

Output: [10, 512, 224, 224] (float32)
  where 512 = 192 + 208 + 48 + 64
```

### Parameters
- Total: 376,176
- branch1x1: 92,352 (Conv2d 480->192)
- branch3x3: 226,096 (Conv2d 480->96: 46,176 + Conv2d 96->208: 179,920)
- branch5x5: 26,944 (Conv2d 480->16: 7,696 + Conv2d 16->48: 19,248)
- branch_pool: 30,784 (Conv2d 480->64: 30,784)

### Config Values
- in_channels = 480
- out_1x1 = 192
- reduce_3x3 = 96, out_3x3 = 208
- reduce_5x5 = 16, out_5x5 = 48
- pool_proj = 64
- batch_size = 10, height = 224, width = 224

## Decomposition Strategy

### Level Classification
- L3 Model: Full inception module (4 parallel branches + cat)
- L2 Layer: Not used -- branches are too simple for layer classification
- L1 Fusion: branch_3x3, branch_5x5, branch_pool (each is 2 adjacent ops)
- L0 Kernel: branch1x1 (single Conv2d), plus all individual ops from fusions

### Decomposition Steps
1. Model (L3) -> 1 kernel (branch1x1) + 3 fusions (branch_3x3, branch_5x5, branch_pool)
2. Each fusion (L1) -> 2 kernels each (reduce/conv or pool/proj)

### Unique L0 Kernels (7 total)
1. conv2d_480x192_1x1_fp32.py -- branch1x1
2. conv2d_480x96_1x1_fp32.py -- branch3x3 reduce
3. conv2d_96x208_3x3_fp32.py -- branch3x3 conv
4. conv2d_480x16_1x1_fp32.py -- branch5x5 reduce
5. conv2d_16x48_5x5_fp32.py -- branch5x5 conv
6. maxpool2d_3x3_fp32.py -- branch_pool pooling
7. conv2d_480x64_1x1_fp32.py -- branch_pool projection

## Verification Summary
- Step 1 (Model->Branches): PASS, max_diff=0.00e+00
- Step 2a (branch3x3->kernels): PASS, max_diff=0.00e+00
- Step 2b (branch5x5->kernels): PASS, max_diff=0.00e+00
- Step 2c (branch_pool->kernels): PASS, max_diff=0.00e+00
- Composition test: PASS, max_diff=0.00e+00
- Coverage: 100% (conv2d: 6/6, max_pool2d: 1/1)
