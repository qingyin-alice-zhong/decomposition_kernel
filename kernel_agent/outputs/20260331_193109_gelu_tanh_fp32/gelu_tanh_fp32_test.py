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


FULL_SHAPE = (1, 1024, 3072)
FULL_ELEMS = int(np.prod(FULL_SHAPE))
TILE_ELEMS = 1024

INPUT_SIZE = TILE_ELEMS
OUTPUT_SIZE = TILE_ELEMS

NUMERIC_RTOL = 1.5e-2
NUMERIC_ATOL = 2.5e-2

Ly = Layout("R")


def build_input_case(case_name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if case_name == "normal":
        x = rng.standard_normal(FULL_SHAPE, dtype=np.float32)
    elif case_name == "uniform_small":
        x = rng.uniform(-0.5, 0.5, size=FULL_SHAPE).astype(np.float32)
    elif case_name == "uniform_large":
        x = rng.uniform(-8.0, 8.0, size=FULL_SHAPE).astype(np.float32)
    elif case_name == "boundary_mix":
        base = np.array([-10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
        reps = (FULL_ELEMS + len(base) - 1) // len(base)
        x = np.tile(base, reps)[:FULL_ELEMS].reshape(FULL_SHAPE)
    else:
        raise ValueError(f"Unknown input case: {case_name}")
    return x.astype(np.float32, copy=False)


def build_reference(x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        ref = F.gelu(torch.from_numpy(x), approximate="tanh").cpu().numpy().astype(np.float32)
    return ref


def _build_top(kernel_path: str):
    gelu_kernel = ExternalModule(
        top="gelu_tanh_fp32",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(A: float32[INPUT_SIZE], C: float32[OUTPUT_SIZE]):
        @df.kernel(mapping=[1], args=[A, C])
        def core(local_A: float32[INPUT_SIZE] @ Ly, local_C: float32[OUTPUT_SIZE] @ Ly):
            gelu_kernel(local_A, local_C)

    return top


def run_tiled_gelu(mod, x: np.ndarray) -> np.ndarray:
    flat_x = x.reshape(-1)
    flat_out = np.zeros((FULL_ELEMS,), dtype=np.float32)

    warm_in = np.zeros((INPUT_SIZE,), dtype=np.float32)
    warm_out = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
    mod(warm_in, warm_out)

    tiles = FULL_ELEMS // TILE_ELEMS
    for t in range(tiles):
        start = t * TILE_ELEMS
        end = start + TILE_ELEMS
        in_tile = flat_x[start:end]
        out_tile = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
        mod(in_tile, out_tile)
        flat_out[start:end] = out_tile

    print(f"Executed {tiles} tiles for gelu_tanh_fp32")
    return flat_out.reshape(FULL_SHAPE)


def compute_error_stats(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(actual - expected)
    return float(np.max(diff)), float(np.mean(diff))


def _test_gelu_tanh_fp32(kernel_path: str):
    print(
        "Running tiled validation for gelu_tanh_fp32: "
        f"shape={FULL_SHAPE}, tile_elems={TILE_ELEMS}"
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
        aggregate_max_abs = 0.0
        aggregate_mean_abs = 0.0
        aggregate_count = 0
        total_cases = 0

        for seed in seeds:
            for case_name in cases:
                x = build_input_case(case_name, seed)
                ref = build_reference(x)
                out = run_tiled_gelu(mod, x)
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
                        out_repeat = run_tiled_gelu(mod, x)
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
    parser.add_argument("--kernel_path", type=str, default="gelu_tanh_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_gelu_tanh_fp32(args.kernel_path)
