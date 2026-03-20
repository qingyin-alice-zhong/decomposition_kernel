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
def reference_shuffle_int32(x: Annotated[np.ndarray, "shape: (256,)"], seed: Annotated[np.ndarray, "shape: (1,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    # Fisher-Yates shuffle with deterministic seed
    arr = x.copy()
    state = np.uint32(seed[0])
    for i in range(len(arr) - 1, 0, -1):
        state ^= (state << np.uint32(13))
        state ^= (state >> np.uint32(17))
        state ^= (state << np.uint32(5))
        j = int(state % np.uint32(i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr.astype(np.int32)
# Reference code ends


def _test_shuffle_int32(kernel_path: str):
    shuffle_int32_kernel = ExternalModule(
        top="shuffle_int32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int32[M] @ Ly, B: int32[256] @ Ly, Param0: int32[1]):
            shuffle_int32_kernel(A, B, Param0)

    input_tensor1 = np.random.randint(-10000, 10000, (256,), dtype=np.int32)
    param0 = np.array([2], dtype=np.int32)
    ref_output = reference_shuffle_int32(input_tensor1, param0)

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
        output_allo = np.zeros((256,), dtype=np.int32)
        mod(input_tensor1, output_allo, param0)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="shuffle_int32", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_shuffle_int32(args.kernel_path)
