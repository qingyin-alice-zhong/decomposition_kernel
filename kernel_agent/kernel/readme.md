# Kernel Workspace (NPU Kernel Debug & Verification)

This directory is used to maintain and iterate operator kernel implementations under `decomposition_workspace/kernel_agent/kernel/*`.
The goal is to make every kernel reach the following on the NPU path:

1. Compilable (`compilation success`)
2. Executable (`execution success`)
3. Numerically verifiable (`verification success`)

---

## Workflow Overview

Each kernel is centered around three files:

- `{kernel_name}.py`: semantic reference (PyTorch/high-level semantics, do not modify)
- `{kernel_name}.cc`: NPU external kernel implementation
- `{kernel_name}_test.py`: build, invoke, assemble, and verification driver

Typical iterative loop:

1. Edit `.cc` / `_test.py`
2. Run `main.py --kernel <name>`
3. Inspect logs and summary under `outputs/<timestamp>_<kernel>/`
4. Continue with minimal changes based on failure evidence

---

## Quick Start

### 1) Run a single kernel

```bash
cd /home/qz425/allo
python ~/decomposition_workspace/kernel_agent/main.py --kernel linear_720x2048_fp32
```

For a longer timeout:

```bash
python ~/decomposition_workspace/kernel_agent/main.py --kernel linear_720x2048_fp32 --timeout 7200
```

### 2) Common verification tiers (recommended)

Standard tier (fast regression):

```bash
ALLO_VERIFY_MODE=full ALLO_SEQ_SAMPLE=0,25,49 ALLO_REPEAT_COUNT=1 \
python ~/decomposition_workspace/kernel_agent/main.py --kernel linear_720x2048_fp32
```

Final tier (high-coverage acceptance):

```bash
ALLO_VERIFY_MODE=full ALLO_STRICT_HEAVY=1 ALLO_SEQ_SAMPLE=0,25,49 ALLO_REPEAT_COUNT=2 \
python ~/decomposition_workspace/kernel_agent/main.py --kernel linear_720x2048_fp32
```

Recommended policy: pass the Standard tier first, then run the Final tier.

---

## Outputs and Pass Criteria

Each run generates:

```text
decomposition_workspace/kernel_agent/outputs/<timestamp>_<kernel_name>/
├── run.log
├── compile_output.txt
├── execution_output.txt
├── <kernel_name>.cc
├── <kernel_name>_test.py
└── <kernel_name>_summary.json
```

Focus on `<kernel_name>_summary.json`:

- `status.compilation_success`
- `status.execution_success`
- `status.verification_success`
- `performance_analysis.*`

The run is considered a pass only when all three status fields are `true`.

---

## Recommended Implementation Pattern (Proven Stable)

For large linear/normalization kernels, prioritize a deploy-first micro-kernel strategy:

- `.cc`: implement a small, stable, deployable micro-kernel (for example `32->1` or `32->32`)
- `_test.py`: host-side tiled scheduling and result assembly (tile over sequence/feature)
- Ensure the NPU path is truly runnable first, then gradually increase coverage and strictness

This pattern effectively reduces compile/runtime instability caused by one-shot large-tile mapping.

---

## Current Kernel List

Classification note:

- `abs_int8` and `add_offset_int8` are early example/demo kernels and were not generated as part of the current SmolVLA kernel generation workflow.
- All other kernels currently listed in this directory are kernels required by SmolVLA.

### Elementwise / Activation

- `abs_int8`
- `add_offset_int8`
- `gelu_tanh_fp32`
- `silu_fp32`
- `softmax_fp32`

### Norm

- `layer_norm_768_fp32`
- `rms_norm_720_fp32`
- `rms_norm_960_fp32`

### Embedding

- `embedding_1024x768_fp32`
- `embedding_49280x960_fp32`

### Conv

- `conv2d_3x64_b1a_fp32`
- `conv2d_3x768_16x16_fp32`

### Linear (shape-specific)

- `linear_32x720_fp32`
- `linear_32x960_fp32`
- `linear_320x320_fp32`
- `linear_720x32_fp32`
- `linear_720x320_fp32`
- `linear_720x720_fp32`
- `linear_720x960_fp32`
- `linear_720x2048_fp32`
- `linear_768x768_fp32`
- `linear_768x3072_fp32`
- `linear_960x320_fp32`
- `linear_960x720_fp32`
- `linear_960x960_fp32`
- `linear_960x2560_fp32`
- `linear_1440x720_fp32`
- `linear_2048x720_fp32`
- `linear_2560x960_fp32`
- `linear_3072x768_fp32`
- `linear_12288x960_fp32`

---

## Directory Structure

```text
decomposition_workspace/kernel_agent/kernel/
├── <kernel_name>/
│   ├── <kernel_name>.py
│   ├── <kernel_name>.cc
│   ├── <kernel_name>_test.py
│   └── prompt.py
├── linear_test_common.py
├── linear_tiled_micro_test_common.py
└── readme.md
```

---

## Development Constraints (Recommended)

- Do not modify `{kernel_name}.py`, the semantic reference file.
- In each iteration, make only minimal changes supported by evidence.
- Prioritize compilation/execution success first, then converge numerical errors.
- Base conclusions on the latest summary and logs under `outputs/`.

