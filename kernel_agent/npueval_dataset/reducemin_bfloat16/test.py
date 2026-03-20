import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16, float32
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 64

# Reference code starts
def reference_reducemin_bfloat16(x: Annotated[np.ndarray, "shape: (64,)"]) -> Annotated[np.ndarray, "shape: (1,)"]:
    return np.array(np.min(x).astype(np.float32))
# Reference code ends


def _test_reducemin_bfloat16(kernel_path: str):
    reducemin_bfloat16_kernel = ExternalModule(
        top="reducemin_bfloat16",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: float32[1] @ Ly):
            reducemin_bfloat16_kernel(A, B)

    input_tensor = np.random.randn(tensor_size).astype(np_bfloat16)
    ref_output = reference_reducemin_bfloat16(input_tensor)

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
        output_allo = np.zeros((1,)).astype(np.float32)
        mod(input_tensor, output_allo)
        try:
            np.testing.assert_allclose(
                output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2
            )
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="reducemin_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_reducemin_bfloat16(args.kernel_path)
