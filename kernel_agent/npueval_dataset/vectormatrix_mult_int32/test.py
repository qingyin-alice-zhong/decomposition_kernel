import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int32
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
def reference_vectormatrix_mult_int32(vector: Annotated[np.ndarray, "shape: (16,)"], matrix_flat: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (16,)"]:
    vector_size = 16
    matrix_cols = 16
    m = matrix_flat.reshape((vector_size, matrix_cols))
    result = np.zeros(matrix_cols, dtype=np.int32)
    for i in range(matrix_cols):
        for j in range(vector_size):
            result[i] += vector[j] * m[j, i]
    return result
# Reference code ends


def _test_vectormatrix_mult_int32(kernel_path: str):
    vectormatrix_mult_int32_kernel = ExternalModule(
        top="vectormatrix_mult_int32",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int32[16] @ Ly, B: int32[M] @ Ly, C: int32[16] @ Ly):
            vectormatrix_mult_int32_kernel(A, B, C)

    vector_tensor = np.random.randint(-100, 100, (16,), dtype=np.int32)   # 16-element vector
    matrix_tensor = np.random.randint(-100, 100, (256,), dtype=np.int32)  # 16x16 flattened
    
    ref_output = reference_vectormatrix_mult_int32(vector_tensor, matrix_tensor)

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
        output_allo = np.zeros((16,), dtype=np.int32)
        mod(vector_tensor, matrix_tensor, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="vectormatrix_mult_int32", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_vectormatrix_mult_int32(args.kernel_path)
