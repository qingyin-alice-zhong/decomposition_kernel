import os
import argparse
import shutil
import math
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
LyRep = [R]

MAPPING_CORES = int(os.environ.get("ALLO_MAPPING_CORES", "4"))

FULL_BATCH = 1
IN_CHANNELS = 3
FULL_OUT_CHANNELS = 768
FULL_INPUT_HEIGHT = 512
FULL_INPUT_WIDTH = 512
FULL_OUTPUT_HEIGHT = 32
FULL_OUTPUT_WIDTH = 32
KERNEL_H = 16
KERNEL_W = 16
STRIDE_H = 16
STRIDE_W = 16

TILE_OUT_CHANNELS = 4
TILE_OUT_HEIGHT = 2
TILE_OUT_WIDTH = 2
TILE_INPUT_HEIGHT = TILE_OUT_HEIGHT * STRIDE_H
TILE_INPUT_WIDTH = TILE_OUT_WIDTH * STRIDE_W

INPUT_SIZE = IN_CHANNELS * TILE_INPUT_HEIGHT * TILE_INPUT_WIDTH
OUTPUT_SIZE = TILE_OUT_CHANNELS * TILE_OUT_HEIGHT * TILE_OUT_WIDTH
WEIGHT_SIZE = TILE_OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W
BIAS_SIZE = TILE_OUT_CHANNELS
PARAM_SIZE = WEIGHT_SIZE + BIAS_SIZE

FULL_RUN = os.environ.get("ALLO_FULL_RUN", "1") == "1"
MAX_GROUPS = int(os.environ.get("ALLO_MAX_GROUPS", "8"))
STRICT_VERIFY = os.environ.get("ALLO_STRICT_VERIFY", "1") == "1"
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


def build_test_case(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    full_input = (rng.standard_normal((FULL_BATCH, IN_CHANNELS, FULL_INPUT_HEIGHT, FULL_INPUT_WIDTH), dtype=np.float32) * 0.05).astype(np.float32)
    full_weight = (rng.standard_normal((FULL_OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), dtype=np.float32) * 0.05).astype(np.float32)
    full_bias = (rng.standard_normal((FULL_OUT_CHANNELS,), dtype=np.float32) * 0.02).astype(np.float32)

    with torch.no_grad():
        ref_full = F.conv2d(
            torch.from_numpy(full_input),
            torch.from_numpy(full_weight),
            torch.from_numpy(full_bias),
            stride=(STRIDE_H, STRIDE_W),
            padding=0,
        ).cpu().numpy()

    return (
        full_input,
        full_weight.astype(np.float32, copy=False),
        full_bias.astype(np.float32, copy=False),
        ref_full.astype(np.float32, copy=False),
    )


def run_tiled_reference(mod, full_input: np.ndarray, full_weight: np.ndarray, full_bias: np.ndarray, ref_output: np.ndarray):
    output = np.zeros((FULL_BATCH, FULL_OUT_CHANNELS, FULL_OUTPUT_HEIGHT, FULL_OUTPUT_WIDTH), dtype=np.float32)

    tile_tasks: list[tuple[int, int, int, int, int, int]] = []
    for n in range(FULL_BATCH):
        for oc_start, oc_extent in iter_tile_starts(FULL_OUT_CHANNELS, TILE_OUT_CHANNELS):
            for h_start, h_extent in iter_tile_starts(FULL_OUTPUT_HEIGHT, TILE_OUT_HEIGHT):
                for w_start, w_extent in iter_tile_starts(FULL_OUTPUT_WIDTH, TILE_OUT_WIDTH):
                    tile_tasks.append((n, oc_start, oc_extent, h_start, h_extent, w_start))

    if not FULL_RUN and MAX_GROUPS > 0:
        tile_tasks = tile_tasks[: MAX_GROUPS * MAPPING_CORES]
        print(
            f"[RUN_MODE] capped run: tiles={len(tile_tasks)}, groups={math.ceil(len(tile_tasks) / MAPPING_CORES)}, cores={MAPPING_CORES}"
        )
    else:
        print(f"[RUN_MODE] full run: tiles={len(tile_tasks)}, cores={MAPPING_CORES}")

    tile_counter = 0
    mismatch_tiles = 0
    mismatch_samples: list[str] = []

    warm_input = np.zeros((INPUT_SIZE,), dtype=np.float32)
    warm_output = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
    warm_param = np.zeros((PARAM_SIZE,), dtype=np.float32)
    mod(
        warm_input, warm_input, warm_input, warm_input,
        warm_output, warm_output, warm_output, warm_output,
        warm_param, warm_param, warm_param, warm_param,
    )

    for group_start in range(0, len(tile_tasks), MAPPING_CORES):
        group = tile_tasks[group_start:group_start + MAPPING_CORES]
        input_group = np.zeros((MAPPING_CORES, INPUT_SIZE), dtype=np.float32)
        param_group = np.zeros((MAPPING_CORES, PARAM_SIZE), dtype=np.float32)

        active_meta: list[tuple[int, int, int, int, int, int, int]] = []
        for lane, task in enumerate(group):
            n, oc_start, oc_extent, h_start, h_extent, w_start = task

            in_h_start = h_start * STRIDE_H
            in_h_end = in_h_start + TILE_INPUT_HEIGHT
            in_w_start = w_start * STRIDE_W
            in_w_end = in_w_start + TILE_INPUT_WIDTH

            input_patch = full_input[n, :, in_h_start:in_h_end, in_w_start:in_w_end]
            if input_patch.shape != (IN_CHANNELS, TILE_INPUT_HEIGHT, TILE_INPUT_WIDTH):
                raise RuntimeError(
                    f"Unexpected input patch shape {input_patch.shape} for tile "
                    f"(n={n}, oc={oc_start}, h={h_start}, w={w_start})"
                )

            weight_patch = np.zeros((TILE_OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), dtype=np.float32)
            bias_patch = np.zeros((TILE_OUT_CHANNELS,), dtype=np.float32)
            weight_patch[:oc_extent] = full_weight[oc_start:oc_start + oc_extent]
            bias_patch[:oc_extent] = full_bias[oc_start:oc_start + oc_extent]

            input_group[lane] = input_patch.reshape(-1)
            param_group[lane] = np.concatenate([weight_patch.reshape(-1), bias_patch.reshape(-1)]).astype(np.float32)
            active_meta.append((lane, n, oc_start, oc_extent, h_start, h_extent, w_start))

        output_group = np.zeros((MAPPING_CORES, OUTPUT_SIZE), dtype=np.float32)
        mod(
            input_group[0], input_group[1], input_group[2], input_group[3],
            output_group[0], output_group[1], output_group[2], output_group[3],
            param_group[0], param_group[1], param_group[2], param_group[3],
        )

        for lane, n, oc_start, oc_extent, h_start, h_extent, w_start in active_meta:
            w_extent = min(TILE_OUT_WIDTH, FULL_OUTPUT_WIDTH - w_start)
            tile_output = output_group[lane].reshape(TILE_OUT_CHANNELS, TILE_OUT_HEIGHT, TILE_OUT_WIDTH)
            npu_tile = tile_output[:oc_extent, :h_extent, :w_extent]
            expected_tile = ref_output[
                n,
                oc_start:oc_start + oc_extent,
                h_start:h_start + h_extent,
                w_start:w_start + w_extent,
            ]

            tile_ok = np.isfinite(npu_tile).all() and np.allclose(
                npu_tile,
                expected_tile,
                rtol=NUMERIC_RTOL,
                atol=NUMERIC_ATOL,
            )
            if not tile_ok:
                mismatch_tiles += 1
                if len(mismatch_samples) < 8:
                    diff = np.abs(npu_tile - expected_tile)
                    mismatch_samples.append(
                        f"tile(n={n}, oc={oc_start}, h={h_start}, w={w_start}) max_abs_diff={float(np.max(diff)):.6f}"
                    )

            output[
                n,
                oc_start:oc_start + oc_extent,
                h_start:h_start + h_extent,
                w_start:w_start + w_extent,
            ] = tile_output[:oc_extent, :h_extent, :w_extent]
            tile_counter += 1

    print(f"Executed {tile_counter} tiles")
    if mismatch_tiles > 0:
        print(f"Detected mismatched tiles: {mismatch_tiles}")
        for sample in mismatch_samples:
            print(f"  - {sample}")

    return output, mismatch_tiles, mismatch_samples


def _print_test_geometry() -> None:
    buffer_bytes = estimate_buffer_bytes()
    tile_count = (
        FULL_BATCH
        * math.ceil(FULL_OUT_CHANNELS / TILE_OUT_CHANNELS)
        * math.ceil(FULL_OUTPUT_HEIGHT / TILE_OUT_HEIGHT)
        * math.ceil(FULL_OUTPUT_WIDTH / TILE_OUT_WIDTH)
    )
    print(
        "Running tiled validation for conv2d_3x768_16x16_fp32: "
        f"N={FULL_BATCH}, C_in={IN_CHANNELS}, C_out={FULL_OUT_CHANNELS}, "
        f"H_in=W_in={FULL_INPUT_HEIGHT}, H_out=W_out={FULL_OUTPUT_HEIGHT}, "
        f"tile=(C_out={TILE_OUT_CHANNELS}, H_out={TILE_OUT_HEIGHT}, W_out={TILE_OUT_WIDTH}), "
        f"num_tiles={tile_count}"
    )
    print(
        "Tile buffers (bytes): "
        f"input={buffer_bytes['input_bytes']}, output={buffer_bytes['output_bytes']}, param={buffer_bytes['param_bytes']}"
    )


def _test_conv2d_3x768_16x16_fp32(kernel_path: str):
    conv_kernel = ExternalModule(
        top="conv2d_3x768_16x16_fp32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    @df.region()
    def top(
        A0: float32[INPUT_SIZE],
        A1: float32[INPUT_SIZE],
        A2: float32[INPUT_SIZE],
        A3: float32[INPUT_SIZE],
        C0: float32[OUTPUT_SIZE],
        C1: float32[OUTPUT_SIZE],
        C2: float32[OUTPUT_SIZE],
        C3: float32[OUTPUT_SIZE],
        Param0: float32[PARAM_SIZE],
        Param1: float32[PARAM_SIZE],
        Param2: float32[PARAM_SIZE],
        Param3: float32[PARAM_SIZE],
    ):
        @df.kernel(
            mapping=[MAPPING_CORES],
            args=[A0, A1, A2, A3, C0, C1, C2, C3, Param0, Param1, Param2, Param3],
        )
        def core(
            local_A0: float32[INPUT_SIZE] @ LyRep,
            local_A1: float32[INPUT_SIZE] @ LyRep,
            local_A2: float32[INPUT_SIZE] @ LyRep,
            local_A3: float32[INPUT_SIZE] @ LyRep,
            local_C0: float32[OUTPUT_SIZE] @ LyRep,
            local_C1: float32[OUTPUT_SIZE] @ LyRep,
            local_C2: float32[OUTPUT_SIZE] @ LyRep,
            local_C3: float32[OUTPUT_SIZE] @ LyRep,
            local_Param0: float32[PARAM_SIZE] @ LyRep,
            local_Param1: float32[PARAM_SIZE] @ LyRep,
            local_Param2: float32[PARAM_SIZE] @ LyRep,
            local_Param3: float32[PARAM_SIZE] @ LyRep,
        ):
            pid, = df.get_pid()
            with allo.meta_if(pid == 0):
                conv_kernel(local_A0, local_C0, local_Param0)
            with allo.meta_elif(pid == 1):
                conv_kernel(local_A1, local_C1, local_Param1)
            with allo.meta_elif(pid == 2):
                conv_kernel(local_A2, local_C2, local_Param2)
            with allo.meta_else():
                conv_kernel(local_A3, local_C3, local_Param3)

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

        output_allo, mismatch_tiles, mismatch_samples = run_tiled_reference(
            mod,
            full_input,
            full_weight,
            full_bias,
            ref_output,
        )

        try:
            if STRICT_VERIFY and mismatch_tiles > 0:
                raise AssertionError(f"strict verification failed: mismatched_tiles={mismatch_tiles}")

            np.testing.assert_allclose(output_allo, ref_output, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
            if mismatch_tiles > 0:
                for sample in mismatch_samples[:8]:
                    print(f"  - {sample}")

        print(f"Mapping used: [{MAPPING_CORES}] (tiled reconstruction mode)")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="conv2d_3x768_16x16_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv2d_3x768_16x16_fp32(args.kernel_path)
