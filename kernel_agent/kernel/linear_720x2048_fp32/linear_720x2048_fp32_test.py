import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch

import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import TOP_PRJ_ABS_DIR


FULL_BATCH = 1
FULL_SEQ = 50
IN_FEATURES = 720
OUT_FEATURES = 2048

KERNEL_IN_FEATURES = 32
OUT_TILE = 1
ABI_TILE = 1

INPUT_SIZE = KERNEL_IN_FEATURES
OUTPUT_SIZE = ABI_TILE
PARAM_SIZE = ABI_TILE * KERNEL_IN_FEATURES

NUMERIC_RTOL = 1e-4
NUMERIC_ATOL = 1e-4

Ly = Layout("R")


def build_case(seed: int, case_name: str):
    rng = np.random.default_rng(seed)

    if case_name == "normal":
        x = rng.standard_normal((FULL_BATCH, FULL_SEQ, IN_FEATURES), dtype=np.float32)
    elif case_name == "uniform_small":
        x = rng.uniform(-0.5, 0.5, size=(FULL_BATCH, FULL_SEQ, IN_FEATURES)).astype(np.float32)
    elif case_name == "uniform_large":
        x = rng.uniform(-5.0, 5.0, size=(FULL_BATCH, FULL_SEQ, IN_FEATURES)).astype(np.float32)
    elif case_name == "boundary_mix":
        base = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], dtype=np.float32)
        total = FULL_BATCH * FULL_SEQ * IN_FEATURES
        reps = (total + len(base) - 1) // len(base)
        x = np.tile(base, reps)[:total].reshape(FULL_BATCH, FULL_SEQ, IN_FEATURES).astype(np.float32)
    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    weight = (rng.standard_normal((OUT_FEATURES, IN_FEATURES), dtype=np.float32) * 0.05).astype(np.float32)

    with torch.no_grad():
        ref = torch.nn.functional.linear(
            torch.from_numpy(x),
            torch.from_numpy(weight),
            bias=None,
        ).cpu().numpy().astype(np.float32)

    return x.astype(np.float32, copy=False), weight, ref


def _build_top(kernel_path: str):
    linear_kernel = ExternalModule(
        top="linear_720x2048_fp32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    @df.region()
    def top(A: float32[INPUT_SIZE], C: float32[OUTPUT_SIZE], P: float32[PARAM_SIZE]):
        @df.kernel(mapping=[1], args=[A, C, P])
        def core(local_A: float32[INPUT_SIZE] @ Ly, local_C: float32[OUTPUT_SIZE] @ Ly, local_P: float32[PARAM_SIZE] @ Ly):
            linear_kernel(local_A, local_C, local_P)

    return top


def run_tiled_linear(mod, x: np.ndarray, weight: np.ndarray):
    output = np.zeros((FULL_BATCH, FULL_SEQ, OUT_FEATURES), dtype=np.float32)
    seq_sample_raw = os.environ.get("ALLO_SEQ_SAMPLE", "0")
    seq_indices = sorted(
        {
            idx
            for idx in (int(v.strip()) for v in seq_sample_raw.split(",") if v.strip())
            if 0 <= idx < FULL_SEQ
        }
    )
    if not seq_indices:
        seq_indices = [0]

    tile_count = 0
    for n in range(FULL_BATCH):
        for s in seq_indices:
            in_vec = np.ascontiguousarray(x[n, s, :], dtype=np.float32)
            for out_start in range(0, OUT_FEATURES, OUT_TILE):
                out_extent = min(OUT_TILE, OUT_FEATURES - out_start)
                partial_sum = np.zeros((OUTPUT_SIZE,), dtype=np.float32)

                for in_start in range(0, IN_FEATURES, KERNEL_IN_FEATURES):
                    in_chunk = np.zeros((KERNEL_IN_FEATURES,), dtype=np.float32)
                    valid = min(KERNEL_IN_FEATURES, IN_FEATURES - in_start)
                    in_chunk[:valid] = in_vec[in_start:in_start + valid]

                    param = np.zeros((PARAM_SIZE,), dtype=np.float32)
                    weight_chunk = np.zeros((ABI_TILE, KERNEL_IN_FEATURES), dtype=np.float32)
                    weight_chunk[:out_extent, :valid] = weight[
                        out_start:out_start + out_extent,
                        in_start:in_start + valid,
                    ]
                    param[:PARAM_SIZE] = weight_chunk.reshape(-1)

                    out_vec = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
                    mod(in_chunk, out_vec, param)
                    partial_sum[:out_extent] += out_vec[:out_extent]
                    tile_count += 1

                output[n, s, out_start:out_start + out_extent] = partial_sum[:out_extent]

    print(
        f"Executed {tile_count} tiles for linear_720x2048_fp32 "
        f"(sampled_seq={seq_indices})"
    )
    return output, seq_indices


def compute_error_stats(actual: np.ndarray, expected: np.ndarray):
    diff = np.abs(actual - expected)
    return float(np.max(diff)), float(np.mean(diff))


def _test(kernel_path: str):
    print(
        "Running tiled validation for linear_720x2048_fp32: "
        f"shape=(1,50,720)->(1,50,2048), out_tile={OUT_TILE}, abi_tile={ABI_TILE}"
    )

    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "basic").lower()
    repeat_count = int(os.environ.get("ALLO_REPEAT_COUNT", "1"))
    strict_heavy = os.environ.get("ALLO_STRICT_HEAVY", "0") == "1"

    if verify_mode == "basic":
        seeds = [0]
        cases = ["normal"]
    else:
        if strict_heavy:
            seeds = [0, 7, 13, 23]
            cases = ["normal", "uniform_small", "uniform_large", "boundary_mix"]
        else:
            seeds = [0, 7]
            cases = ["normal", "uniform_small"]

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
        return

    top = _build_top(kernel_path)
    mod = df.build(
        top,
        target="aie",
        profile=False,
        warmup=0,
        num_iters=1,
        project=TOP_PRJ_ABS_DIR,
    )

    failures = []
    total_cases = 0
    aggregate_max_abs = 0.0
    aggregate_mean_abs = 0.0
    aggregate_count = 0

    for seed in seeds:
        for case_name in cases:
            x, weight, ref = build_case(seed=seed, case_name=case_name)
            out, seq_indices = run_tiled_linear(mod, x, weight)
            total_cases += 1

            out_eval = out[:, seq_indices, :]
            ref_eval = ref[:, seq_indices, :]

            if not np.isfinite(out_eval).all():
                failures.append(f"seed={seed}, case={case_name}: output contains non-finite values")
                continue

            try:
                np.testing.assert_allclose(out_eval, ref_eval, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
            except AssertionError as e:
                failures.append(f"seed={seed}, case={case_name}: {str(e)}")
                continue

            max_abs, mean_abs = compute_error_stats(out_eval, ref_eval)
            aggregate_max_abs = max(aggregate_max_abs, max_abs)
            aggregate_mean_abs += mean_abs
            aggregate_count += 1

            if seed == seeds[0] and case_name == cases[0] and repeat_count > 1:
                stable = True
                for rep in range(repeat_count - 1):
                    out_repeat, _ = run_tiled_linear(mod, x, weight)
                    out_repeat_eval = out_repeat[:, seq_indices, :]
                    if not np.allclose(out_repeat_eval, out_eval, rtol=0.0, atol=0.0):
                        failures.append(
                            f"seed={seed}, case={case_name}: repeated run #{rep + 2} produced different outputs"
                        )
                        stable = False
                        break
                    try:
                        np.testing.assert_allclose(out_repeat_eval, ref_eval, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
                    except AssertionError as e:
                        failures.append(
                            f"seed={seed}, case={case_name}: repeated run #{rep + 2} mismatch: {str(e)}"
                        )
                        stable = False
                        break
                if not stable:
                    continue

            print(
                f"[CASE PASS] seed={seed}, case={case_name}, "
                f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
            )

    if aggregate_count > 0:
        print(
            f"Aggregate error stats: max_abs={aggregate_max_abs:.3e}, "
            f"mean_abs={aggregate_mean_abs / aggregate_count:.3e} over {aggregate_count} cases"
        )

    if failures:
        print("FAIL!")
        print(f"Verification failed: {len(failures)}/{total_cases} cases failed")
        for msg in failures[:8]:
            print(f"  - {msg}")
    elif total_cases == 0 or aggregate_count != total_cases:
        print("FAIL!")
        print(
            "Verification bookkeeping mismatch: "
            f"total_cases={total_cases}, aggregate_count={aggregate_count}"
        )
    else:
        print(f"All verification cases passed: {total_cases}/{total_cases}")
        print("PASS!")

    print("Trace analysis skipped (profile=False).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="linear_720x2048_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test(args.kernel_path)
