import os
import argparse
import shutil
import math
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
MAPPING_CORES = 4

FULL_BATCH = 10
IN_CHANNELS = 3
FULL_OUT_CHANNELS = 64
FULL_HEIGHT = 224
FULL_WIDTH = 224
KERNEL_H = 3
KERNEL_W = 3
PADDING = 1

TILE_OUT_CHANNELS = 8
TILE_OUT_HEIGHT = 16
TILE_OUT_WIDTH = 16
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


def iter_tile_starts(full_extent: int, tile_extent: int):
    for start in range(0, full_extent, tile_extent):
        yield start, min(tile_extent, full_extent - start)


def direct_reference_from_tile(
    input_flat: np.ndarray,
    param: np.ndarray,
    out_channels: int,
    out_height: int,
    out_width: int,
) -> np.ndarray:
    dynamic_weight_size = out_channels * IN_CHANNELS * KERNEL_H * KERNEL_W
    weights = param[:dynamic_weight_size].reshape(out_channels, IN_CHANNELS, KERNEL_H, KERNEL_W)
    bias = param[dynamic_weight_size:dynamic_weight_size + out_channels]
    input_patch = input_flat.reshape(IN_CHANNELS, TILE_INPUT_HEIGHT, TILE_INPUT_WIDTH)
    output = np.zeros((out_channels, out_height, out_width), dtype=np.float32)

    for oc in range(out_channels):
        for oh in range(out_height):
            for ow in range(out_width):
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
    spatial_size = FULL_HEIGHT * FULL_WIDTH
    n = flat_idx // (FULL_OUT_CHANNELS * spatial_size)
    rem0 = flat_idx % (FULL_OUT_CHANNELS * spatial_size)
    oc = rem0 // spatial_size
    rem1 = rem0 % spatial_size
    oh = rem1 // FULL_WIDTH
    ow = rem1 % FULL_WIDTH
    return (
        f"max_abs_diff={max_diff:.8f} at flat_idx={flat_idx} "
        f"(n={n}, oc={oc}, oh={oh}, ow={ow}), actual={float(actual[flat_idx]):.8f}, "
        f"expected={float(expected[flat_idx]):.8f}"
    )


def build_test_case(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    full_input = (rng.standard_normal((FULL_BATCH, IN_CHANNELS, FULL_HEIGHT, FULL_WIDTH), dtype=np.float32) * 0.05).astype(np.float32)
    full_weight = (rng.standard_normal((FULL_OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), dtype=np.float32) * 0.05).astype(np.float32)
    full_bias = (rng.standard_normal((FULL_OUT_CHANNELS,), dtype=np.float32) * 0.02).astype(np.float32)

    with torch.no_grad():
        ref_full = F.conv2d(
            torch.from_numpy(full_input),
            torch.from_numpy(full_weight),
            torch.from_numpy(full_bias),
            stride=1,
            padding=PADDING,
        ).cpu().numpy()

    return (
        full_input,
        full_weight.astype(np.float32, copy=False),
        full_bias.astype(np.float32, copy=False),
        ref_full.astype(np.float32, copy=False),
    )


def run_tiled_reference(
    mod,
    full_input: np.ndarray,
    full_weight: np.ndarray,
    full_bias: np.ndarray,
) -> np.ndarray:
    padded_input = np.pad(
        full_input,
        ((0, 0), (0, 0), (PADDING, PADDING), (PADDING, PADDING)),
        mode="constant",
        constant_values=0.0,
    )
    output = np.zeros((FULL_BATCH, FULL_OUT_CHANNELS, FULL_HEIGHT, FULL_WIDTH), dtype=np.float32)

    total_tiles = (
        FULL_BATCH
        * math.ceil(FULL_OUT_CHANNELS / TILE_OUT_CHANNELS)
        * math.ceil(FULL_HEIGHT / TILE_OUT_HEIGHT)
        * math.ceil(FULL_WIDTH / TILE_OUT_WIDTH)
    )
    tile_counter = 0

    for n in range(FULL_BATCH):
        for oc_start, oc_extent in iter_tile_starts(FULL_OUT_CHANNELS, TILE_OUT_CHANNELS):
            for h_start, h_extent in iter_tile_starts(FULL_HEIGHT, TILE_OUT_HEIGHT):
                for w_start, w_extent in iter_tile_starts(FULL_WIDTH, TILE_OUT_WIDTH):
                    tile_counter += 1

                    in_h_start = h_start
                    in_h_end = h_start + h_extent + 2 * PADDING
                    in_w_start = w_start
                    in_w_end = w_start + w_extent + 2 * PADDING

                    input_patch = padded_input[n, :, in_h_start:in_h_end, in_w_start:in_w_end]
                    if input_patch.shape != (IN_CHANNELS, TILE_INPUT_HEIGHT, TILE_INPUT_WIDTH):
                        raise RuntimeError(
                            f"Unexpected input patch shape {input_patch.shape} for tile "
                            f"(n={n}, oc={oc_start}, h={h_start}, w={w_start})"
                        )

                    weight_patch = np.zeros((TILE_OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), dtype=np.float32)
                    bias_patch = np.zeros((TILE_OUT_CHANNELS,), dtype=np.float32)
                    weight_patch[:oc_extent] = full_weight[oc_start:oc_start + oc_extent]
                    bias_patch[:oc_extent] = full_bias[oc_start:oc_start + oc_extent]

                    param = np.concatenate([weight_patch.reshape(-1), bias_patch.reshape(-1)]).astype(np.float32)
                    input_flat = input_patch.reshape(-1).astype(np.float32)
                    output_flat = np.zeros((OUTPUT_SIZE,), dtype=np.float32)

                    mod(input_flat, output_flat, param)

                    tile_output = output_flat.reshape(TILE_OUT_CHANNELS, TILE_OUT_HEIGHT, TILE_OUT_WIDTH)
                    output[
                        n,
                        oc_start:oc_start + oc_extent,
                        h_start:h_start + h_extent,
                        w_start:w_start + w_extent,
                    ] = tile_output[:oc_extent, :h_extent, :w_extent]

    print(f"Executed {tile_counter}/{total_tiles} tiles for full-shape reconstruction")
    return output


def _print_test_geometry() -> None:
    buffer_bytes = estimate_buffer_bytes()
    tile_count = (
        FULL_BATCH
        * math.ceil(FULL_OUT_CHANNELS / TILE_OUT_CHANNELS)
        * math.ceil(FULL_HEIGHT / TILE_OUT_HEIGHT)
        * math.ceil(FULL_WIDTH / TILE_OUT_WIDTH)
    )
    print(
        "Running full-shape tiled validation for conv2d_3x64_b1a_fp32: "
        f"N={FULL_BATCH}, C_in={IN_CHANNELS}, C_out={FULL_OUT_CHANNELS}, H=W={FULL_HEIGHT}, "
        f"tile=(C_out={TILE_OUT_CHANNELS}, H={TILE_OUT_HEIGHT}, W={TILE_OUT_WIDTH}), "
        f"num_tiles={tile_count}"
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
        @df.kernel(mapping=[MAPPING_CORES])
        def core(A: float32[972] @ Ly, C: float32[2048] @ Ly, Param: float32[224] @ Ly):
            conv_kernel(A, C, Param)

    _print_test_geometry()
    full_input, full_weight, full_bias, ref_output = build_test_case(seed=0)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=False,
            warmup=0,
            num_iters=1,
            project=TOP_PRJ_ABS_DIR,
        )
        output_allo = run_tiled_reference(mod, full_input, full_weight, full_bias)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-4, atol=1e-4)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
            print(describe_mismatch(output_allo.reshape(-1), ref_output.reshape(-1)))

        print(f"Mapping used: [{MAPPING_CORES}] (full tiled reconstruction mode)")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv2d_3x64_b1a_fp32(args.kernel_path)
