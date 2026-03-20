import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16, int8
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
def reference_cast_bfloat16_to_int8(input_vector: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    # Convert bfloat16 to float32 for processing
    x = input_vector.astype(np.float32)
    xi = np.trunc(x).astype(np.int32)
    floor_x = np.where(
        x >= 0,
        xi,
        np.where(x == xi, xi, xi - 1)
    )
    frac = x - floor_x.astype(np.float32)
    even_floor = (floor_x & 1) == 0
    r = np.where(
        frac < 0.5,
        floor_x,
        np.where(
            frac > 0.5,
            floor_x + 1,
            np.where(even_floor, floor_x, floor_x + 1)
        )
    )
    r = np.clip(r, -128, 127)
    return r.astype(np.int8)
# Reference code ends


def _test_cast_bfloat16_to_int8(kernel_path: str):
    cast_bfloat16_to_int8_kernel = ExternalModule(
        top="cast_bfloat16_to_int8",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: int8[M] @ Ly):
            cast_bfloat16_to_int8_kernel(A, B)

    # Generate random bfloat16 input in a reasonable range
    input_tensor = (np.random.randn(256) * 70).astype(np_bfloat16)
    
    ref_output = reference_cast_bfloat16_to_int8(input_tensor)

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
        output_allo = np.zeros((256,), dtype=np.int8)
        mod(input_tensor, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="cast_bfloat16_to_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_cast_bfloat16_to_int8(args.kernel_path)
