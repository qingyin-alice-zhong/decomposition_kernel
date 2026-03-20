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
def reference_pad1d_int32(input_vector: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (2,)"]) -> Annotated[np.ndarray, "shape: (260,)"]:
    pad_size_val = param[0]
    pad_value_val = param[1]
    padded_length = len(input_vector) + 2 * pad_size_val
    padded_vector = np.full(padded_length, pad_value_val, dtype=np.int32)
    padded_vector[pad_size_val:pad_size_val + len(input_vector)] = input_vector
    return padded_vector
# Reference code ends


def _test_pad1d_int32(kernel_path: str):
    pad1d_int32_kernel = ExternalModule(
        top="pad1d_int32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int32[M] @ Ly, B: int32[260] @ Ly, Param: int32[2] @ Ly):
            pad1d_int32_kernel(A, B, Param)

    input_vector = np.random.randint(-10, 10, (256,), dtype=np.int32)
    pad_size = 2
    pad_value = 0
    param = np.array([pad_size, pad_value], dtype=np.int32)
    ref_output = reference_pad1d_int32(input_vector, param)

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
        output_allo = np.zeros((260,), dtype=np.int32)
        mod(input_vector, output_allo, param)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="pad1d_int32", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_pad1d_int32(args.kernel_path)
