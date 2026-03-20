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
tensor_size = 256  # 16x16 input
output_size = 400  # 20x20 output

# Reference code starts
def reference_pad2d_int32(input_flat: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (2,)"]) -> Annotated[np.ndarray, "shape: (400,)"]:
    # Unpack parameters: pad_size, pad_value
    pad_size = int(param[0])
    pad_value = int(param[1])
    # Reshape flat input to 16x16 matrix
    input_matrix = input_flat.reshape((16, 16))
    input_rows, input_cols = input_matrix.shape
    padded_rows = input_rows + 2 * pad_size
    padded_cols = input_cols + 2 * pad_size
    padded_matrix = np.full((padded_rows, padded_cols), pad_value, dtype=input_matrix.dtype)
    padded_matrix[pad_size:pad_size + input_rows, pad_size:pad_size + input_cols] = input_matrix
    return padded_matrix.flatten().astype(np.int32)
# Reference code ends


def _test_pad2d_int32(kernel_path: str):
    pad2d_int32_kernel = ExternalModule(
        top="pad2d_int32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int32[M] @ Ly, B: int32[400] @ Ly, Param: int32[2] @ Ly):
            pad2d_int32_kernel(A, B, Param)

    input_tensor = np.random.randint(-10, 10, (256,), dtype=np.int32)  # 16x16 input
    pad_size = 2
    pad_value = 0
    # Combine pad_size and pad_value into single parameter array
    param = np.array([pad_size, pad_value], dtype=np.int32)
    ref_output = reference_pad2d_int32(input_tensor, param)

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
        output_allo = np.zeros((400,), dtype=np.int32)
        mod(input_tensor, output_allo, param)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="pad2d_int32", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_pad2d_int32(args.kernel_path)
