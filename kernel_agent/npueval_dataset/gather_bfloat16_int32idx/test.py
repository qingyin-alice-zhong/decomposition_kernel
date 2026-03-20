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
tensor_size = 256

# Reference code starts
def reference_gather_bfloat16_int32idx(input_data: Annotated[np.ndarray, "shape: (256,)"], indices: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    return input_data[indices].astype(np_bfloat16)
# Reference code ends


def _test_gather_bfloat16_int32idx(kernel_path: str):
    gather_bfloat16_int32idx_kernel = ExternalModule(
        top="gather_bfloat16_int32idx",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: int32[M] @ Ly, C: bfloat16[M] @ Ly):
            gather_bfloat16_int32idx_kernel(A, B, C)



    # Generate random bfloat16 input data and int32 indices
    src_buffer_size = 256
    input_data = (np.random.randn(src_buffer_size)).astype(np_bfloat16)
    indices = np.random.randint(0, src_buffer_size, size=src_buffer_size).astype(np.int32)
    
    ref_output = reference_gather_bfloat16_int32idx(input_data, indices)

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
        output_allo = np.zeros((tensor_size,), dtype=np_bfloat16)
        mod(input_data, indices, output_allo)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="gather_bfloat16_int32idx", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_gather_bfloat16_int32idx(args.kernel_path)
