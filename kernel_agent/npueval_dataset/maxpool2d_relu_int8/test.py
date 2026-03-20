import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int8
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 256  # 16x16 input flattened

# Reference code starts
def reference_maxpool2d_relu_int8(input_flat: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (64,)"]:
    # Reshape flat input to 16x16
    input_2d = input_flat.reshape((16, 16))
    rows, cols = input_2d.shape
    pooled = np.zeros((rows // 2, cols // 2), dtype=input_2d.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input_2d[i:i+2, j:j+2]
            max_val = np.max(window)
            pooled[i//2, j//2] = max(max_val, 0)  # Apply ReLU
    return pooled.flatten().astype(np.int8)
# Reference code ends


def _test_maxpool2d_relu_int8(kernel_path: str):
    maxpool2d_relu_int8_kernel = ExternalModule(
        top="maxpool2d_relu_int8",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = int8
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int8[M] @ Ly, B: int8[64] @ Ly):
            maxpool2d_relu_int8_kernel(A, B)

    input_tensor = np.random.randint(-128, 128, (256,), dtype=np.int8)  # 16x16 flattened
    
    ref_output = reference_maxpool2d_relu_int8(input_tensor)

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
        output_allo = np.zeros((64,), dtype=np.int8)
        mod(input_tensor, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="maxpool2d_relu_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_maxpool2d_relu_int8(args.kernel_path)
