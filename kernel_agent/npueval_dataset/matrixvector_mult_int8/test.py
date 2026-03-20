import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int8, int32
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
def reference_matrixvector_mult_int8(matrix_flat: Annotated[np.ndarray, "shape: (256,)"], vector: Annotated[np.ndarray, "shape: (16,)"]) -> Annotated[np.ndarray, "shape: (16,)"]:
    # Reshape flat matrix to 16x16
    mat = matrix_flat.reshape((16, 16)).astype(np.int32)
    vec = vector.astype(np.int32)
    result = np.zeros(16, dtype=np.int32)
    for i in range(16):
        acc = 0
        for j in range(16):
            acc += mat[i, j] * vec[j]
        result[i] = acc
    return result
# Reference code ends


def _test_matrixvector_mult_int8(kernel_path: str):
    matrixvector_mult_int8_kernel = ExternalModule(
        top="matrixvector_mult_int8",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int8
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int8[M] @ Ly, B: int8[16] @ Ly, C: int32[16] @ Ly):
            matrixvector_mult_int8_kernel(A, B, C)

    matrix_tensor = np.random.randint(-20, 20, (256,), dtype=np.int8)  # 16x16 flattened
    vector_tensor = np.random.randint(-20, 20, (16,), dtype=np.int8)    # 16-element vector
    
    ref_output = reference_matrixvector_mult_int8(matrix_tensor, vector_tensor)

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
        mod(matrix_tensor, vector_tensor, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="matrixvector_mult_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_matrixvector_mult_int8(args.kernel_path)
