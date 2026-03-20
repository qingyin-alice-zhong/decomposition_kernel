import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int32, int32
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
def reference_conv1d_int32(in_buffer: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (4,)"]) -> Annotated[np.ndarray, "shape: (254,)"]:
    vector_size = 256
    kernel_size = 3
    kernel = param[:3]  # First 3 elements are kernel
    stride_val = param[3]  # Last element is stride
    num_windows = (vector_size - kernel_size) // stride_val + 1
    
    out_buffer = np.zeros(num_windows, dtype=np.int32)
    for i in range(num_windows):
        conv_sum = 0
        for j in range(kernel_size):
            conv_sum += in_buffer[i * stride_val + j] * kernel[j]
        out_buffer[i] = conv_sum
    
    return out_buffer
# Reference code ends


def _test_conv1d_int32(kernel_path: str):
    conv1d_int32_kernel = ExternalModule(
        top="conv1d_int32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int32[M] @ Ly, C: int32[254] @ Ly, Param: int32[4] @ Ly):
            conv1d_int32_kernel(A, C, Param)

    input_tensor = np.random.randint(-100, 100, (256,), dtype=np.int32)
    kernel_tensor = np.random.randint(-10, 10, (3,), dtype=np.int32)
    stride_val = 1
    # Combine kernel and stride into single parameter array
    param = np.concatenate([kernel_tensor, [stride_val]]).astype(np.int32)
    ref_output = reference_conv1d_int32(input_tensor, param)

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
        output_allo = np.zeros((254,), dtype=np.int32)
        mod(input_tensor, output_allo, param)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv1d_int32", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv1d_int32(args.kernel_path)
