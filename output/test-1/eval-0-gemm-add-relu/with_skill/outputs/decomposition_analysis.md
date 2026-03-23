# Decomposition Analysis: 76_Gemm_Add_ReLU

## Source
- File: `data/kernelbench/level2/76_Gemm_Add_ReLU.py`
- Source level: level2 (KernelBench)

## Architecture Analysis

### Module Hierarchy
- Model (L1 Fusion: Gemm_Add_ReLU)
  - self.gemm: nn.Linear(8192, 8192, bias=False)
  - self.bias: nn.Parameter(shape=(8192,))
  - torch.relu activation (functional call)

### Data Flow (with shapes)
```
Input: [1024, 8192] (float32)
  -> gemm (nn.Linear, no bias): [1024, 8192] -> [1024, 8192]
  -> bias add (+self.bias): [1024, 8192] -> [1024, 8192]
  -> relu: [1024, 8192] -> [1024, 8192]
Output: [1024, 8192] (float32)
```

### Dimensions (from original model)
- batch_size = 1024
- in_features = 8192
- out_features = 8192
- bias_shape = (8192,)

## Abstraction Level Classification

The model is classified as **Level 1 (Fusion)** because:
1. It performs 3 adjacent operations (Linear + BiasAdd + ReLU) in a straight line
2. No residual connections, skip connections, or data flow branches
3. All operations feed directly into the next
4. This is a classic fusible pattern (GEMM + elementwise ops)

## Decomposition Plan

### Step 1: Fusion (L1) -> Kernels (L0)
- **linear_8192x8192_nobias_fp32.py**: nn.Linear(8192, 8192, bias=False)
- **bias_add_8192_fp32.py**: Learnable bias parameter addition
- **relu_fp32.py**: ReLU activation (shape-independent elementwise)

## Kernel Inventory

| Kernel File | Operation | Parameters | Instances |
|---|---|---|---|
| linear_8192x8192_nobias_fp32.py | Linear (no bias) | weight: (8192, 8192) | 1 |
| bias_add_8192_fp32.py | Bias Add | bias: (8192,) | 1 |
| relu_fp32.py | ReLU | (none) | 1 |

## Verification Results

- Step 1 verify_step.py: PASS (max_diff=0.00e+00)
- Composition test: PASS (max_diff=0.00e+00)
- Coverage: 100% (2/2 compute op types covered)
