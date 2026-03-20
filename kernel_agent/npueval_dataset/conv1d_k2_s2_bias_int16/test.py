import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int16, int32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 128

# Reference code starts
def reference_conv1d_k2_s2_bias_int16(input_buffer: Annotated[np.ndarray, "shape: (128,)"], param: Annotated[np.ndarray, "shape: (3,)"]) -> Annotated[np.ndarray, "shape: (64,)"]:
    # Unpack parameters: kernel[2] + bias[1] = param[3]
    kernel = param[:2]
    bias = param[2:3]
    vector_size = input_buffer.shape[0]
    kernel_size = kernel.shape[0]
    stride = 2
    num_windows = (vector_size - kernel_size) // stride + 1
    out = np.zeros(num_windows, dtype=np.int16)
    for i in range(num_windows):
        acc = int(bias[0])
        for j in range(kernel_size):
            acc += int(input_buffer[i * stride + j]) * int(kernel[j])
        acc = np.clip(acc, -32768, 32767)
        out[i] = acc
    return out.astype(np.int16)
# Reference code ends


def _test_conv1d_k2_s2_bias_int16(kernel_path: str):
    conv1d_k2_s2_bias_int16_kernel = ExternalModule(
        top="conv1d_k2_s2_bias_int16",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int16[M] @ Ly, C: int16[64] @ Ly, Param: int32[3] @ Ly):
            conv1d_k2_s2_bias_int16_kernel(A, C, Param)

    input_tensor1 = np.random.randint(-1000, 1000, (128,), dtype=np.int16)
    kernel_tensor = np.random.randint(-10, 10, (2,), dtype=np.int16)
    bias_value = 2
    # Combine kernel and bias into single parameter array
    param = np.array([kernel_tensor[0], kernel_tensor[1], bias_value], dtype=np.int32)
    ref_output = reference_conv1d_k2_s2_bias_int16(input_tensor1, param)

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
        output_allo = np.zeros((64,), dtype=np.int16)
        mod(input_tensor1, output_allo, param)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv1d_k2_s2_bias_int16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv1d_k2_s2_bias_int16(args.kernel_path)
