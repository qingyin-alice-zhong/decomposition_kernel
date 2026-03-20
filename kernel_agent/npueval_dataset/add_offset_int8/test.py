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
def reference_add_offset_int8(x: Annotated[np.ndarray, "shape: (256,)"], offset: Annotated[np.ndarray, "shape: (1,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    res = x.astype(np.int16) + int(offset[0])
    res = np.clip(res, -128, 127)
    return res.astype(np.int8)
# Reference code ends


def _test_add_offset_int8(kernel_path: str):
    add_kernel = ExternalModule(
        top="add_offset_int8",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int8
    Ty_scalar = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, Off: Ty_scalar[1]):
            add_kernel(A, B, Off)

    input_tensor = np.random.randint(-100, 100, (tensor_size,), dtype=np.int8)
    offset = np.random.randint(-40, 40, (1,), dtype=np.int32)
    ref_output = reference_add_offset_int8(input_tensor, offset)

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
        output_allo = np.zeros((tensor_size,), dtype=np.int8)
        mod(input_tensor, output_allo, offset)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="add_offset_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_add_offset_int8(args.kernel_path)




