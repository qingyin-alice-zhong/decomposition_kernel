import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16, int32
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 256

# Reference code starts
def reference_conv1d_k4_s1_bias_relu_bfloat16(in_buffer: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (5,)"]) -> Annotated[np.ndarray, "shape: (252,)"]:
    # Extract kernel[4] and bias from param (5 float32 values packed as int32)
    kernel_vals = np.array([param[i] for i in range(4)], dtype=np.int32).view(np.float32)
    bias_val = np.array([param[4]], dtype=np.int32).view(np.float32)[0]
    
    kernel = kernel_vals.astype(np_bfloat16)
    bias = np_bfloat16(bias_val)
    
    vector_size = in_buffer.shape[0]
    kernel_size = 4
    stride = 1
    output_size = min((vector_size - kernel_size) // stride + 1, 252)  # Limit to 252 for 4-byte alignment
    out_buffer = np.zeros(output_size, dtype=np_bfloat16)
    for i in range(output_size):
        acc = 0.0
        for j in range(kernel_size):
            acc += float(in_buffer[i * stride + j]) * float(kernel[j])
        acc += float(bias)
        out_buffer[i] = np_bfloat16(max(0.0, acc))  # ReLU
    return out_buffer
# Reference code ends


def _test_conv1d_k4_s1_bias_relu_bfloat16(kernel_path: str):
    conv1d_k4_s1_bias_relu_bfloat16_kernel = ExternalModule(
        top="conv1d_k4_s1_bias_relu_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, C: bfloat16[252] @ Ly, Param: int32[5] @ Ly):
            conv1d_k4_s1_bias_relu_bfloat16_kernel(A, C, Param)

    # Generate random bfloat16 input and parameters
    in_buffer = (np.random.randn(256) * 3.2).astype(np_bfloat16)
    
    # Pack kernel[4] and bias into param array (5 float32 values as int32)
    kernel_vals = (np.random.randn(4) * 0.5).astype(np.float32)
    bias_val = np.random.randn() * 0.1
    
    param = np.array([
        kernel_vals[0].view(np.int32), kernel_vals[1].view(np.int32),
        kernel_vals[2].view(np.int32), kernel_vals[3].view(np.int32),
        np.float32(bias_val).view(np.int32)
    ], dtype=np.int32)
    
    ref_output = reference_conv1d_k4_s1_bias_relu_bfloat16(in_buffer, param.astype(np.float32))

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            warmup=5,
            num_iters=20,
            trace=[("core", (0,))],
            trace_size=655360,
            project=TOP_PRJ_ABS_DIR
        )
        output_allo = np.zeros((252,), dtype=np_bfloat16)
        mod(in_buffer, output_allo, param)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=3e-2, atol=3e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv1d_k4_s1_bias_relu_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv1d_k4_s1_bias_relu_bfloat16(args.kernel_path)
