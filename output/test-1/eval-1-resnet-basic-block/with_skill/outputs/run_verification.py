#!/usr/bin/env python3
"""
Run all verification steps for the ResNet BasicBlock decomposition.

Usage: python3 run_verification.py
Run from the decomposition_workspace root directory.
"""

import subprocess
import sys
import os
from pathlib import Path

# Set working directory to workspace root
ws_root = Path(__file__).parent.parent.parent.parent.parent
os.chdir(ws_root)

output_dir = Path("decompose-workspace/iteration-1/eval-1-resnet-basic-block/with_skill/outputs")
original_model = Path("data/kernelbench/level3/8_ResNetBasicBlock.py")

steps = [
    {
        "name": "Step 1: Layer -> Fusions",
        "original": output_dir / "steps/step_1_model_to_fusions/original.py",
        "refactored": output_dir / "steps/step_1_model_to_fusions/refactored.py",
        "output": output_dir / "steps/step_1_model_to_fusions/verification_result.json",
    },
    {
        "name": "Step 2a: Conv-BN-ReLU -> Kernels",
        "original": output_dir / "steps/step_2_conv_bn_relu_to_kernels/original.py",
        "refactored": output_dir / "steps/step_2_conv_bn_relu_to_kernels/refactored.py",
        "output": output_dir / "steps/step_2_conv_bn_relu_to_kernels/verification_result.json",
    },
    {
        "name": "Step 2b: Conv-BN -> Kernels",
        "original": output_dir / "steps/step_2_conv_bn_to_kernels/original.py",
        "refactored": output_dir / "steps/step_2_conv_bn_to_kernels/refactored.py",
        "output": output_dir / "steps/step_2_conv_bn_to_kernels/verification_result.json",
    },
    {
        "name": "Step 2c: Downsample -> Kernels",
        "original": output_dir / "steps/step_2_downsample_to_kernels/original.py",
        "refactored": output_dir / "steps/step_2_downsample_to_kernels/refactored.py",
        "output": output_dir / "steps/step_2_downsample_to_kernels/verification_result.json",
    },
]

all_pass = True

print("=" * 70)
print("RESNET BASICBLOCK DECOMPOSITION VERIFICATION")
print("=" * 70)
print()

# Run each verification step
for step in steps:
    print(f"\n--- {step['name']} ---")
    cmd = [
        sys.executable, "scripts/verify_step.py",
        "--original", str(step["original"]),
        "--refactored", str(step["refactored"]),
        "--output", str(step["output"]),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        all_pass = False
        print(f"  FAILED (exit code {result.returncode})")
    print()

# Run composition test
print("\n--- Composition Test ---")
comp_cmd = [sys.executable, str(output_dir / "verification/composition_test.py")]
result = subprocess.run(comp_cmd, capture_output=False)
if result.returncode != 0:
    all_pass = False
    print("  Composition test FAILED")
print()

# Run coverage analysis
print("\n--- Coverage Analysis ---")
cov_cmd = [
    sys.executable, "scripts/extract_ops.py",
    "--model", str(original_model),
    "--decomp-dir", str(output_dir),
    "--output", str(output_dir / "verification/coverage_summary.json"),
]
result = subprocess.run(cov_cmd, capture_output=False)
if result.returncode != 0:
    print("  Coverage analysis returned non-zero (may be partial coverage)")
print()

# Final summary
print("=" * 70)
if all_pass:
    print("ALL VERIFICATION STEPS PASSED")
else:
    print("SOME VERIFICATION STEPS FAILED - see above for details")
print("=" * 70)

sys.exit(0 if all_pass else 1)
