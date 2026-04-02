import argparse
import os
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
from utils import TOP_PRJ_ABS_DIR


FULL_BATCH = 1
FULL_SEQ = 1024
FULL_HIDDEN = 768
FULL_SHAPE = (FULL_BATCH, FULL_SEQ, FULL_HIDDEN)

INPUT_SIZE = FULL_HIDDEN
OUTPUT_SIZE = FULL_HIDDEN
PARAM_SIZE = FULL_HIDDEN * 2 + 2

EPS = 1e-6
NUMERIC_RTOL = 3e-2
NUMERIC_ATOL = 3e-2

Ly = Layout("R")


def build_case(seed: int, case_name: str):
    rng = np.random.default_rng(seed)

    if case_name == "normal":
        x = rng.standard_normal(FULL_SHAPE, dtype=np.float32)
    elif case_name == "uniform_small":
        x = rng.uniform(-0.5, 0.5, size=FULL_SHAPE).astype(np.float32)
    elif case_name == "uniform_large":
        x = rng.uniform(-10.0, 10.0, size=FULL_SHAPE).astype(np.float32)
    elif case_name == "boundary_mix":
        base = np.array([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0], dtype=np.float32)
        reps = (np.prod(FULL_SHAPE) + len(base) - 1) // len(base)
        x = np.tile(base, reps)[: np.prod(FULL_SHAPE)].reshape(FULL_SHAPE).astype(np.float32)
    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    gamma = (rng.uniform(0.8, 1.2, size=(FULL_HIDDEN,))).astype(np.float32)
    beta = (rng.uniform(-0.1, 0.1, size=(FULL_HIDDEN,))).astype(np.float32)

    with torch.no_grad():
        ref = F.layer_norm(
            torch.from_numpy(x),
            normalized_shape=(FULL_HIDDEN,),
            weight=torch.from_numpy(gamma),
            bias=torch.from_numpy(beta),
            eps=EPS,
        ).cpu().numpy().astype(np.float32)

    return x, gamma, beta, ref


def _build_top(kernel_path: str):
    ln_kernel = ExternalModule(
        top="layer_norm_768_fp32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    @df.region()
    def top(A: float32[INPUT_SIZE], C: float32[OUTPUT_SIZE], P: float32[PARAM_SIZE]):
        @df.kernel(mapping=[1], args=[A, C, P])
        def core(local_A: float32[INPUT_SIZE] @ Ly, local_C: float32[OUTPUT_SIZE] @ Ly, local_P: float32[PARAM_SIZE] @ Ly):
            ln_kernel(local_A, local_C, local_P)

    return top


def run_tiled_layer_norm(mod, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    output = np.zeros_like(x, dtype=np.float32)

    param = np.zeros((PARAM_SIZE,), dtype=np.float32)
    param[:FULL_HIDDEN] = gamma
    param[FULL_HIDDEN:2 * FULL_HIDDEN] = beta
    param[2 * FULL_HIDDEN] = EPS

    warm_in = np.zeros((INPUT_SIZE,), dtype=np.float32)
    warm_out = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
    warm_param = np.zeros((PARAM_SIZE,), dtype=np.float32)
    mod(warm_in, warm_out, warm_param)

    tile_count = 0
    for n in range(FULL_BATCH):
        for s in range(FULL_SEQ):
            in_tile = x[n, s, :].astype(np.float32, copy=False)
            out_tile = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
            mod(in_tile, out_tile, param)
            output[n, s, :] = out_tile
            tile_count += 1

    print(f"Executed {tile_count} tiles for layer_norm_768_fp32")
    return output


def compute_error_stats(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(actual - expected)
    return float(np.max(diff)), float(np.mean(diff))


def _test_layer_norm_768_fp32(kernel_path: str):
    print(
        "Running tiled validation for layer_norm_768_fp32: "
        f"shape={FULL_SHAPE}, hidden={FULL_HIDDEN}"
    )

    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "full").lower()
    repeat_count = int(os.environ.get("ALLO_REPEAT_COUNT", "2"))

    if verify_mode == "basic":
        seeds = [0]
        cases = ["normal"]
    else:
        seeds = [0, 7, 13]
        cases = ["normal", "uniform_small", "uniform_large", "boundary_mix"]

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        top = _build_top(kernel_path)
        mod = df.build(
            top,
            target="aie",
            profile=False,
            warmup=0,
            num_iters=1,
            project=TOP_PRJ_ABS_DIR,
        )

        failures: list[str] = []
        total_cases = 0
        aggregate_max_abs = 0.0
        aggregate_mean_abs = 0.0
        aggregate_count = 0

        for seed in seeds:
            for case_name in cases:
                x, gamma, beta, ref = build_case(seed=seed, case_name=case_name)
                out = run_tiled_layer_norm(mod, x, gamma, beta)
                total_cases += 1

                if not np.isfinite(out).all():
                    failures.append(f"seed={seed}, case={case_name}: output contains non-finite values")
                    continue

                try:
                    np.testing.assert_allclose(out, ref, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
                except AssertionError as e:
                    failures.append(f"seed={seed}, case={case_name}: {str(e)}")
                    continue

                max_abs, mean_abs = compute_error_stats(out, ref)
                aggregate_max_abs = max(aggregate_max_abs, max_abs)
                aggregate_mean_abs += mean_abs
                aggregate_count += 1

                if seed == seeds[0] and case_name == cases[0] and repeat_count > 1:
                    stable = True
                    for rep in range(repeat_count - 1):
                        out_repeat = run_tiled_layer_norm(mod, x, gamma, beta)
                        if not np.allclose(out_repeat, out, rtol=0.0, atol=0.0):
                            failures.append(
                                f"seed={seed}, case={case_name}: repeated run #{rep + 2} produced different outputs"
                            )
                            stable = False
                            break
                        try:
                            np.testing.assert_allclose(out_repeat, ref, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
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
        else:
            print(f"All verification cases passed: {total_cases}/{total_cases}")
            print("PASS!")

        print("Trace analysis skipped (profile=False).")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="layer_norm_768_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_layer_norm_768_fp32(args.kernel_path)
