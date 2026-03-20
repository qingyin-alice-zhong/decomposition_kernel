import os
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")

FULL_BATCH = 10
IN_CHANNELS = 3
FULL_OUT_CHANNELS = 64
FULL_HEIGHT = 224
FULL_WIDTH = 224
KERNEL_H = 3
KERNEL_W = 3
PADDING = 1

# Default test mode validates a deterministic slice of the original VGG conv.
# This keeps the generated AIE buffers within tile memory while still checking
# correctness against the full PyTorch operator on the corresponding region.
TILE_BATCH_INDEX = 0
TILE_OUT_CHANNEL_OFFSET = 0
TILE_OUT_CHANNELS = 8
TILE_OUT_HEIGHT = 16
TILE_OUT_WIDTH = 16
TILE_BASE_H = 32
TILE_BASE_W = 48
TILE_INPUT_HEIGHT = TILE_OUT_HEIGHT + KERNEL_H - 1
TILE_INPUT_WIDTH = TILE_OUT_WIDTH + KERNEL_W - 1

INPUT_SIZE = IN_CHANNELS * TILE_INPUT_HEIGHT * TILE_INPUT_WIDTH
OUTPUT_SIZE = TILE_OUT_CHANNELS * TILE_OUT_HEIGHT * TILE_OUT_WIDTH
WEIGHT_SIZE = TILE_OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W
BIAS_SIZE = TILE_OUT_CHANNELS
PARAM_SIZE = WEIGHT_SIZE + BIAS_SIZE


def estimate_buffer_bytes() -> dict[str, int]:
    return {
        "input_bytes": INPUT_SIZE * 4,
        "output_bytes": OUTPUT_SIZE * 4,
        "param_bytes": PARAM_SIZE * 4,
    }


def direct_reference_from_tile(input_flat: np.ndarray, param: np.ndarray) -> np.ndarray:
    weights = param[:WEIGHT_SIZE].reshape(TILE_OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W)
    bias = param[WEIGHT_SIZE:]
    input_patch = input_flat.reshape(IN_CHANNELS, TILE_INPUT_HEIGHT, TILE_INPUT_WIDTH)
    output = np.zeros((TILE_OUT_CHANNELS, TILE_OUT_HEIGHT, TILE_OUT_WIDTH), dtype=np.float32)

    for oc in range(TILE_OUT_CHANNELS):
        for oh in range(TILE_OUT_HEIGHT):
            for ow in range(TILE_OUT_WIDTH):
                acc = bias[oc]
                for ic in range(IN_CHANNELS):
                    for kh in range(KERNEL_H):
                        for kw in range(KERNEL_W):
                            acc += input_patch[ic, oh + kh, ow + kw] * weights[oc, ic, kh, kw]
                output[oc, oh, ow] = acc

    return output.reshape(-1)


def describe_mismatch(actual: np.ndarray, expected: np.ndarray) -> str:
    diff = np.abs(actual - expected)
    flat_idx = int(np.argmax(diff))
    max_diff = float(diff[flat_idx])
    oc = flat_idx // (TILE_OUT_HEIGHT * TILE_OUT_WIDTH)
    rem = flat_idx % (TILE_OUT_HEIGHT * TILE_OUT_WIDTH)
    oh = rem // TILE_OUT_WIDTH
    ow = rem % TILE_OUT_WIDTH
    return (
        f"max_abs_diff={max_diff:.8f} at flat_idx={flat_idx} "
        f"(oc={oc}, oh={oh}, ow={ow}), actual={float(actual[flat_idx]):.8f}, "
        f"expected={float(expected[flat_idx]):.8f}"
    )


def build_test_case(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    full_input = (rng.standard_normal((FULL_BATCH, IN_CHANNELS, FULL_HEIGHT, FULL_WIDTH), dtype=np.float32) * 0.05).astype(np.float32)
    full_weight = (rng.standard_normal((FULL_OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), dtype=np.float32) * 0.05).astype(np.float32)
    full_bias = (rng.standard_normal((FULL_OUT_CHANNELS,), dtype=np.float32) * 0.02).astype(np.float32)

    in_h_start = TILE_BASE_H - PADDING
    in_h_end = TILE_BASE_H + TILE_OUT_HEIGHT + PADDING
    in_w_start = TILE_BASE_W - PADDING
    in_w_end = TILE_BASE_W + TILE_OUT_WIDTH + PADDING

    input_patch = full_input[TILE_BATCH_INDEX, :, in_h_start:in_h_end, in_w_start:in_w_end].copy()
    weight_patch = full_weight[TILE_OUT_CHANNEL_OFFSET:TILE_OUT_CHANNEL_OFFSET + TILE_OUT_CHANNELS].copy()
    bias_patch = full_bias[TILE_OUT_CHANNEL_OFFSET:TILE_OUT_CHANNEL_OFFSET + TILE_OUT_CHANNELS].copy()

    with torch.no_grad():
        ref_full = F.conv2d(
            torch.from_numpy(full_input),
            torch.from_numpy(full_weight),
            torch.from_numpy(full_bias),
            stride=1,
            padding=PADDING,
        ).cpu().numpy()

    ref_tile = ref_full[
        TILE_BATCH_INDEX,
        TILE_OUT_CHANNEL_OFFSET:TILE_OUT_CHANNEL_OFFSET + TILE_OUT_CHANNELS,
        TILE_BASE_H:TILE_BASE_H + TILE_OUT_HEIGHT,
        TILE_BASE_W:TILE_BASE_W + TILE_OUT_WIDTH,
    ].copy()

    param = np.concatenate([
        weight_patch.reshape(-1),
        bias_patch.reshape(-1),
    ]).astype(np.float32)

    direct_ref = direct_reference_from_tile(input_patch.reshape(-1), param)
    np.testing.assert_allclose(direct_ref, ref_tile.reshape(-1), rtol=1e-5, atol=1e-5)

    return input_patch.reshape(-1), param, ref_tile.reshape(-1).astype(np.float32, copy=False)


def _print_test_geometry() -> None:
    buffer_bytes = estimate_buffer_bytes()
    print(
        "Running tiled validation for conv2d_3x64_b1a_fp32: "
        f"batch={TILE_BATCH_INDEX}, out_channels=[{TILE_OUT_CHANNEL_OFFSET}, {TILE_OUT_CHANNEL_OFFSET + TILE_OUT_CHANNELS}), "
        f"spatial=[{TILE_BASE_H}:{TILE_BASE_H + TILE_OUT_HEIGHT}, {TILE_BASE_W}:{TILE_BASE_W + TILE_OUT_WIDTH})"
    )
    print(
        "Tile buffers (bytes): "
        f"input={buffer_bytes['input_bytes']}, output={buffer_bytes['output_bytes']}, param={buffer_bytes['param_bytes']}"
    )



def _test_conv2d_3x64_b1a_fp32(kernel_path: str):
    conv_kernel = ExternalModule(
        top="conv2d_3x64_b1a_fp32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: float32[972] @ Ly, C: float32[2048] @ Ly, Param: float32[224] @ Ly):
            conv_kernel(A, C, Param)

    _print_test_geometry()
    input_tensor, param, ref_output = build_test_case(seed=0)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            warmup=5,
            num_iters=20,
            trace=[("core", (0,))],
            trace_size=262144,
            project=TOP_PRJ_ABS_DIR,
        )
        output_allo = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
        mod(input_tensor, output_allo, param)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-4, atol=1e-4)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
            print(describe_mismatch(output_allo, ref_output))

        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv2d_3x64_b1a_fp32", colshift=1)
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv2d_3x64_b1a_fp32(args.kernel_path)
