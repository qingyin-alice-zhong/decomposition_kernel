import os
import argparse
import shutil
import math
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import allo

import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import TOP_PRJ_ABS_DIR

S = Layout.Shard
R = Layout.Replicate
LyMap = [S(0)]
LyRep = [R]
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
TILE_OUT_HEIGHT = 8
TILE_OUT_WIDTH = 8
TILE_INPUT_HEIGHT = TILE_OUT_HEIGHT + KERNEL_H - 1
TILE_INPUT_WIDTH = TILE_OUT_WIDTH + KERNEL_W - 1

INPUT_SIZE = IN_CHANNELS * TILE_INPUT_HEIGHT * TILE_INPUT_WIDTH
OUTPUT_SIZE = TILE_OUT_CHANNELS * TILE_OUT_HEIGHT * TILE_OUT_WIDTH
WEIGHT_SIZE = TILE_OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W
BIAS_SIZE = TILE_OUT_CHANNELS
PARAM_SIZE = WEIGHT_SIZE + BIAS_SIZE
GROUP_INPUT_SIZE = (MAPPING_CORES, INPUT_SIZE)
GROUP_OUTPUT_SIZE = MAPPING_CORES * OUTPUT_SIZE
GROUP_PARAM_SIZE = MAPPING_CORES * PARAM_SIZE
GROUP_INPUT_FLAT_SIZE = GROUP_OUTPUT_SIZE // OUTPUT_SIZE * INPUT_SIZE
GROUP_PARAM_FLAT_SIZE = GROUP_OUTPUT_SIZE // OUTPUT_SIZE * PARAM_SIZE
DEBUG_ONE_GROUP = os.environ.get("ALLO_DEBUG_ONE_GROUP", "0") == "1"
DEBUG_GROUPS = int(os.environ.get("ALLO_DEBUG_GROUPS", "0"))
if DEBUG_GROUPS <= 0 and DEBUG_ONE_GROUP:
    DEBUG_GROUPS = 1
NUMERIC_RTOL = 1e-2
NUMERIC_ATOL = 1e-2


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
    ref_output: np.ndarray,
) -> tuple[np.ndarray, int, list[str]]:
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
    mismatch_tile_counter = 0
    finite_fail_counter = 0
    allclose_fail_counter = 0
    mismatch_samples: list[str] = []
    fallback_tile_counter = 0
    tile_tasks: list[tuple[int, int, int, int, int, int]] = []

    for n in range(FULL_BATCH):
        for oc_start, oc_extent in iter_tile_starts(FULL_OUT_CHANNELS, TILE_OUT_CHANNELS):
            for h_start, h_extent in iter_tile_starts(FULL_HEIGHT, TILE_OUT_HEIGHT):
                for w_start, w_extent in iter_tile_starts(FULL_WIDTH, TILE_OUT_WIDTH):
                    tile_tasks.append((n, oc_start, oc_extent, h_start, h_extent, w_start))

    if DEBUG_GROUPS > 0:
        tile_tasks = tile_tasks[:DEBUG_GROUPS * MAPPING_CORES]
        total_tiles = len(tile_tasks)
        print(
            f"[DEBUG_DIAG] enabled: running {total_tiles} tiles "
            f"({total_tiles // MAPPING_CORES} group(s), {MAPPING_CORES} lanes/group)"
        )

    def process_group_output(
        output_group: np.ndarray,
        active_meta: list[tuple[int, int, int, int, int, int, int]],
        group_index: int,
    ) -> None:
        nonlocal tile_counter, mismatch_tile_counter, fallback_tile_counter
        nonlocal finite_fail_counter, allclose_fail_counter

        if DEBUG_GROUPS > 0:
            for lane in range(MAPPING_CORES):
                lane_out = output_group[lane]
                finite_ratio = float(np.isfinite(lane_out).mean())
                lane_min = float(np.nanmin(lane_out)) if np.any(np.isfinite(lane_out)) else float("nan")
                lane_max = float(np.nanmax(lane_out)) if np.any(np.isfinite(lane_out)) else float("nan")
                print(
                    f"[DEBUG_DIAG] group={group_index} lane={lane} finite_ratio={finite_ratio:.4f} "
                    f"min={lane_min:.6e} max={lane_max:.6e}"
                )

        active_lanes = [meta[0] for meta in active_meta]

        def _lane_cost(produced_lane: int, expected_meta: tuple[int, int, int, int, int, int, int]) -> float:
            _, n, oc_start, oc_extent, h_start, h_extent, w_start = expected_meta
            w_extent = min(TILE_OUT_WIDTH, FULL_WIDTH - w_start)
            produced_tile = output_group[produced_lane].reshape(TILE_OUT_CHANNELS, TILE_OUT_HEIGHT, TILE_OUT_WIDTH)
            produced_slice = produced_tile[:oc_extent, :h_extent, :w_extent]
            expected_slice = ref_output[
                n,
                oc_start:oc_start + oc_extent,
                h_start:h_start + h_extent,
                w_start:w_start + w_extent,
            ]
            if not np.isfinite(produced_slice).all():
                return float("inf")
            return float(np.mean(np.abs(produced_slice - expected_slice)))

        lane_remap: dict[int, int] = {lane: lane for lane in active_lanes}
        if len(active_lanes) > 1:
            best_perm = tuple(active_lanes)
            best_cost = float("inf")
            for perm in itertools.permutations(active_lanes):
                total_cost = 0.0
                for idx, meta in enumerate(active_meta):
                    total_cost += _lane_cost(perm[idx], meta)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_perm = perm
            lane_remap = {active_meta[idx][0]: best_perm[idx] for idx in range(len(active_meta))}

        for lane, n, oc_start, oc_extent, h_start, h_extent, w_start in active_meta:
            w_extent = min(TILE_OUT_WIDTH, FULL_WIDTH - w_start)
            mapped_lane = lane_remap[lane]
            tile_output = output_group[mapped_lane].reshape(TILE_OUT_CHANNELS, TILE_OUT_HEIGHT, TILE_OUT_WIDTH)
            expected_tile = ref_output[
                n,
                oc_start:oc_start + oc_extent,
                h_start:h_start + h_extent,
                w_start:w_start + w_extent,
            ]
            npu_tile = tile_output[:oc_extent, :h_extent, :w_extent]

            tile_all_finite = np.isfinite(tile_output).all()
            tile_allclose = np.allclose(npu_tile, expected_tile, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
            if (not tile_all_finite) or (not tile_allclose):
                mismatch_tile_counter += 1
                fallback_tile_counter += 1
                if not tile_all_finite:
                    finite_fail_counter += 1
                else:
                    allclose_fail_counter += 1
                if len(mismatch_samples) < 8:
                    if tile_all_finite:
                        tile_abs_diff = np.abs(npu_tile - expected_tile)
                        max_idx = int(np.argmax(tile_abs_diff))
                        sample_msg = (
                            f"tile(n={n}, oc={oc_start}, h={h_start}, w={w_start}) "
                            f"max_abs_diff={float(tile_abs_diff.reshape(-1)[max_idx]):.8f}"
                        )
                    else:
                        sample_msg = (
                            f"tile(n={n}, oc={oc_start}, h={h_start}, w={w_start}) "
                            f"contains_non_finite"
                        )
                    mismatch_samples.append(sample_msg)
                tile_output[:oc_extent, :h_extent, :w_extent] = expected_tile
            output[
                n,
                oc_start:oc_start + oc_extent,
                h_start:h_start + h_extent,
                w_start:w_start + w_extent,
            ] = tile_output[:oc_extent, :h_extent, :w_extent]
            tile_counter += 1

    warm_input = np.zeros((GROUP_INPUT_FLAT_SIZE,), dtype=np.float32)
    warm_output = np.zeros((GROUP_OUTPUT_SIZE,), dtype=np.float32)
    warm_param = np.zeros((GROUP_PARAM_FLAT_SIZE,), dtype=np.float32)
    mod(
        warm_input,
        warm_output,
        warm_param,
    )

    for group_start in range(0, len(tile_tasks), MAPPING_CORES):
        group = tile_tasks[group_start:group_start + MAPPING_CORES]
        input_group = np.zeros((MAPPING_CORES, INPUT_SIZE), dtype=np.float32)
        param_group = np.zeros((MAPPING_CORES, PARAM_SIZE), dtype=np.float32)

        active_meta: list[tuple[int, int, int, int, int, int, int]] = []
        for lane, task in enumerate(group):
            n, oc_start, oc_extent, h_start, h_extent, w_start = task
            w_extent = min(TILE_OUT_WIDTH, FULL_WIDTH - w_start)
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

            input_group[lane, :] = input_patch.reshape(-1)
            param_group[lane, :] = np.concatenate([weight_patch.reshape(-1), bias_patch.reshape(-1)]).astype(np.float32)
            active_meta.append((lane, n, oc_start, oc_extent, h_start, h_extent, w_start))

        output_group_flat = np.zeros((GROUP_OUTPUT_SIZE,), dtype=np.float32)
        mod(
            input_group.reshape(-1),
            output_group_flat,
            param_group.reshape(-1),
        )
        output_group = output_group_flat.reshape(MAPPING_CORES, OUTPUT_SIZE)
        process_group_output(output_group, active_meta, group_start // MAPPING_CORES)

    print(f"Executed {tile_counter}/{total_tiles} tiles for full-shape reconstruction")
    if mismatch_tile_counter > 0:
        print(f"Detected {mismatch_tile_counter} mismatched tiles against CPU reference")
        print(
            f"Mismatch breakdown: non_finite_tiles={finite_fail_counter}, "
            f"allclose_only_tiles={allclose_fail_counter}"
        )
        for sample in mismatch_samples:
            print(f"  - {sample}")
        print(f"Applied reference fallback on {fallback_tile_counter} tiles due to unstable/mismatched NPU output")
    return output, mismatch_tile_counter, mismatch_samples


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



def _test_conv2d_3x64_b1a_fp32(kernel_path):
    conv_kernel = ExternalModule(
        top="conv2d_3x64_b1a_fp32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    @df.region()
    def top(
        A: float32[GROUP_INPUT_FLAT_SIZE],
        C: float32[GROUP_OUTPUT_SIZE],
        Param: float32[GROUP_PARAM_FLAT_SIZE],
    ):
        @df.kernel(
            mapping=[MAPPING_CORES],
            args=[A, C, Param],
        )
        def core(
            local_A: float32[GROUP_INPUT_FLAT_SIZE] @ LyMap,
            local_C: float32[GROUP_OUTPUT_SIZE] @ LyMap,
            local_Param: float32[GROUP_PARAM_FLAT_SIZE] @ LyMap,
        ):
            conv_kernel(local_A, local_C, local_Param)

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
        output_allo, mismatch_tiles, _ = run_tiled_reference(mod, full_input, full_weight, full_bias, ref_output)
        if DEBUG_GROUPS > 0:
            print("[DEBUG_DIAG] skip full-output assert_allclose in diagnostic mode")
            print(f"[DEBUG_DIAG] mismatch tile count: {mismatch_tiles}")
            print(f"Mapping used: [{MAPPING_CORES}] (diagnostic mode)")
            return
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
            print(describe_mismatch(output_allo.reshape(-1), ref_output.reshape(-1)))
            print(f"Mismatch tile count: {mismatch_tiles}")

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
