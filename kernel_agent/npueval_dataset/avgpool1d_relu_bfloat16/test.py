import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16
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
def reference_avgpool1d_relu_bfloat16(in_buffer: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (64,)"]:
    window_size = 4
    stride = 4
    length = in_buffer.shape[0]
    output_size = length // stride  # 256 // 4 = 64
    out = np.zeros(output_size, dtype=np_bfloat16)
    for i in range(output_size):
        window = in_buffer[i * stride : i * stride + window_size].astype(np.float32)
        avg = window.mean()
        relu_avg = np.maximum(avg, 0.0)
        out[i] = relu_avg.astype(np_bfloat16)
    return out
# Reference code ends


def _test_avgpool1d_relu_bfloat16(kernel_path: str):
    avgpool1d_relu_bfloat16_kernel = ExternalModule(
        top="avgpool1d_relu_bfloat16",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: bfloat16[64] @ Ly):
            avgpool1d_relu_bfloat16_kernel(A, B)

    # Generate random bfloat16 input
    input_tensor = (np.random.randn(256) * 3).astype(np_bfloat16)
    
    ref_output = reference_avgpool1d_relu_bfloat16(input_tensor)

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
        output_allo = np.zeros((64,), dtype=np_bfloat16)
        mod(input_tensor, output_allo)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=3e-2, atol=3e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="avgpool1d_relu_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_avgpool1d_relu_bfloat16(args.kernel_path)
