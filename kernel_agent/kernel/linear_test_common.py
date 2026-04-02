import os
from pathlib import Path

import numpy as np
import torch

import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import TOP_PRJ_ABS_DIR


Ly = Layout("R")


def _build_case(batch: int, seq: int, in_features: int, out_features: int, bias: bool, seed: int, case_name: str):
    rng = np.random.default_rng(seed)

    shape = (batch, seq, in_features)
    if case_name == "normal":
        x = rng.standard_normal(shape, dtype=np.float32)
    elif case_name == "uniform_small":
        x = rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
    elif case_name == "uniform_large":
        x = rng.uniform(-5.0, 5.0, size=shape).astype(np.float32)
    elif case_name == "boundary_mix":
        base = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], dtype=np.float32)
        total = batch * seq * in_features
        reps = (total + len(base) - 1) // len(base)
        x = np.tile(base, reps)[:total].reshape(shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    weight = (rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.05).astype(np.float32)
    bias_vec = (rng.standard_normal((out_features,), dtype=np.float32) * 0.02).astype(np.float32) if bias else None

    with torch.no_grad():
        ref = torch.nn.functional.linear(
            torch.from_numpy(x),
            torch.from_numpy(weight),
            torch.from_numpy(bias_vec) if bias else None,
        ).cpu().numpy().astype(np.float32)

    return x, weight, bias_vec, ref


def _build_top(kernel_name: str, kernel_path: str, input_size: int, output_size: int, param_size: int):
    linear_kernel = ExternalModule(
        top=kernel_name,
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    src = f'''
@df.region()
def top(A: float32[{input_size}], C: float32[{output_size}], P: float32[{param_size}]):
    @df.kernel(mapping=[1], args=[A, C, P])
    def core(local_A: float32[{input_size}] @ Ly, local_C: float32[{output_size}] @ Ly, local_P: float32[{param_size}] @ Ly):
        linear_kernel(local_A, local_C, local_P)
'''
    ns = {
        "df": df,
        "float32": float32,
        "Ly": Ly,
        "linear_kernel": linear_kernel,
    }
    exec(src, ns)
    return ns["top"]


def run_linear_kernel_test(
    kernel_name: str,
    kernel_path: str,
    in_features: int,
    out_features: int,
    seq: int,
    out_tile: int,
    has_bias: bool,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    batch = 1
    input_size = in_features
    output_size = out_tile
    param_size = out_tile * in_features + (out_tile if has_bias else 0)

    print(
        f"Running strict validation for {kernel_name}: "
        f"shape=({batch},{seq},{in_features})->({batch},{seq},{out_features}), out_tile={out_tile}"
    )

    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "full").lower()
    repeat_count = int(os.environ.get("ALLO_REPEAT_COUNT", "2"))

    if verify_mode == "basic":
        seeds = [0]
        cases = ["normal"]
    else:
        seeds = [0, 7, 13]
        cases = ["normal", "uniform_small", "uniform_large", "boundary_mix"]

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
        return

    top = _build_top(kernel_name, kernel_path, input_size, output_size, param_size)
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
            x, weight, bias_vec, ref = _build_case(
                batch=batch,
                seq=seq,
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
                seed=seed,
                case_name=case_name,
            )

            out = np.zeros((batch, seq, out_features), dtype=np.float32)
            tile_calls = 0

            for n in range(batch):
                for s in range(seq):
                    in_vec = x[n, s, :].astype(np.float32, copy=False)
                    for o0 in range(0, out_features, out_tile):
                        o1 = min(o0 + out_tile, out_features)
                        cur_tile = o1 - o0

                        out_vec = np.zeros((output_size,), dtype=np.float32)
                        param = np.zeros((param_size,), dtype=np.float32)

                        w_tile = weight[o0:o1, :]
                        param[:cur_tile * in_features] = w_tile.reshape(-1)
                        if has_bias:
                            param[out_tile * in_features : out_tile * in_features + cur_tile] = bias_vec[o0:o1]

                        mod(in_vec, out_vec, param)
                        out[n, s, o0:o1] = out_vec[:cur_tile]
                        tile_calls += 1

            print(f"[TILES] seed={seed}, case={case_name}, calls={tile_calls}")
            total_cases += 1

            if not np.isfinite(out).all():
                failures.append(f"seed={seed}, case={case_name}: output contains non-finite values")
                continue

            try:
                np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)
            except AssertionError as e:
                failures.append(f"seed={seed}, case={case_name}: {str(e)}")
                continue

            diff = np.abs(out - ref)
            max_abs = float(np.max(diff))
            mean_abs = float(np.mean(diff))
            aggregate_max_abs = max(aggregate_max_abs, max_abs)
            aggregate_mean_abs += mean_abs
            aggregate_count += 1

            if seed == seeds[0] and case_name == cases[0] and repeat_count > 1:
                for rep in range(repeat_count - 1):
                    out_repeat = np.zeros((batch, seq, out_features), dtype=np.float32)
                    for n in range(batch):
                        for s in range(seq):
                            in_vec = x[n, s, :].astype(np.float32, copy=False)
                            for o0 in range(0, out_features, out_tile):
                                o1 = min(o0 + out_tile, out_features)
                                cur_tile = o1 - o0
                                out_vec = np.zeros((output_size,), dtype=np.float32)
                                param = np.zeros((param_size,), dtype=np.float32)
                                w_tile = weight[o0:o1, :]
                                param[:cur_tile * in_features] = w_tile.reshape(-1)
                                if has_bias:
                                    param[out_tile * in_features : out_tile * in_features + cur_tile] = bias_vec[o0:o1]
                                mod(in_vec, out_vec, param)
                                out_repeat[n, s, o0:o1] = out_vec[:cur_tile]

                    if not np.allclose(out_repeat, out, rtol=0.0, atol=0.0):
                        failures.append(
                            f"seed={seed}, case={case_name}: repeated run #{rep + 2} produced different outputs"
                        )
                        break

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
