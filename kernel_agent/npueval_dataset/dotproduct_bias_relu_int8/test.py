import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int8, int32, int32
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
def reference_dotproduct_bias_relu_int8(in0: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (257,)"]) -> Annotated[np.ndarray, "shape: (1,)"]:
    # Unpack parameters: in1[256] + bias[1] = param[257]
    in1 = param[:256].astype(np.int8)
    bias = param[256]
    acc = np.dot(in0.astype(np.int32), in1.astype(np.int32)) + int(bias)
    acc = max(acc, 0)
    acc = np.clip(acc, -128, 127)
    return np.array([acc], dtype=np.int32)
# Reference code ends


def _test_dotproduct_bias_relu_int8(kernel_path: str):
    dotproduct_bias_relu_int8_kernel = ExternalModule(
        top="dotproduct_bias_relu_int8",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = int8
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int8[M] @ Ly, C: int32[1] @ Ly, Param: int32[257] @ Ly):
            dotproduct_bias_relu_int8_kernel(A, C, Param)

    input_tensor1 = np.random.randint(-20, 20, (tensor_size,), dtype=np.int8)
    input_tensor2 = np.random.randint(-20, 20, (tensor_size,), dtype=np.int8)
    bias_value = 5
    # Combine input_tensor2 and bias into single parameter array
    param = np.concatenate([input_tensor2.astype(np.int32), np.array([bias_value], dtype=np.int32)])
    ref_output = reference_dotproduct_bias_relu_int8(input_tensor1, param)

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
        output_allo = np.zeros((1,), dtype=np.int32)
        mod(input_tensor1, output_allo, param)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="dotproduct_bias_relu_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_dotproduct_bias_relu_int8(args.kernel_path)
