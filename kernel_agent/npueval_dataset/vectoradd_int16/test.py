import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int16
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
def reference_vectoradd_int16(x1: Annotated[np.ndarray, "shape: (256,)"], x2: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    return (x1 + x2).astype(np.int16)
# Reference code ends


def _test_vectoradd_int16(kernel_path: str):
    add_kernel = ExternalModule(
        top="vectoradd_int16",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, C: Ty[M] @ Ly):
            add_kernel(A, B, C)

    input_tensor1 = np.random.randint(-512, 512, (tensor_size,), dtype=np.int16)
    input_tensor2 = np.random.randint(-512, 512, (tensor_size,), dtype=np.int16)
    ref_output = reference_vectoradd_int16(input_tensor1, input_tensor2)

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
        output_allo = np.zeros((tensor_size,), dtype=np.int16)
        mod(input_tensor1, input_tensor2, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="vectoradd_int16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_vectoradd_int16(args.kernel_path)


