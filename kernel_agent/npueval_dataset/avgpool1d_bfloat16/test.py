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
input_size = 1024
output_size = 512  # (1024 - 2) // 2 + 1 with window_size=2, stride=2

# Reference code starts
def reference_avgpool1d_bfloat16(x: Annotated[np.ndarray, "shape: (1024,)"], 
                                param: Annotated[np.ndarray, "shape: (2,)"]) -> Annotated[np.ndarray, "shape: (512,)"]:
    ws = int(param[0])
    st = int(param[1])
    output_length = (len(x) - ws) // st + 1
    out_buffer = np.zeros(output_length, dtype=np_bfloat16)
    output_idx = 0
    for i in range(0, len(x) - ws + 1, st):
        window = x[i:i+ws]
        out_buffer[output_idx] = np.mean(window)
        output_idx += 1
    return out_buffer.astype(np.float32)
# Reference code ends


def _test_avgpool1d_bfloat16(kernel_path: str):
    avgpool_kernel = ExternalModule(
        top="avgpool1d_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = bfloat16
    Ty_param = int32

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[input_size] @ Ly, B: Ty[output_size] @ Ly, 
                 Param: Ty_param[2] @ Ly):
            avgpool_kernel(A, B, Param)

    input_tensor = (np.random.randn(input_size) * 2).astype(np_bfloat16)
    window_size = 2
    stride = 2
    param = np.array([window_size, stride], dtype=np.int32)
    ref_output = reference_avgpool1d_bfloat16(input_tensor, param)

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
        output_allo = np.zeros((output_size,)).astype(np_bfloat16)
        mod(input_tensor, output_allo, param)
        try:
            np.testing.assert_allclose(
                output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2
            )
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="avgpool1d_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_avgpool1d_bfloat16(args.kernel_path)
