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


BATCH = 1
SEQ = 50
HIDDEN = 720
EPS = 1e-5
CHUNK = 32
RTOL = 1e-4
ATOL = 1e-4
Ly = Layout("R")


def _build_case(seed: int, case_name: str):
    rng = np.random.default_rng(seed)
    shape = (BATCH, SEQ, HIDDEN)
    if case_name == "normal":
        x = rng.standard_normal(shape, dtype=np.float32)
    elif case_name == "uniform_small":
        x = rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
    elif case_name == "uniform_large":
        x = rng.uniform(-8.0, 8.0, size=shape).astype(np.float32)
    else:
        base = np.array([-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0], dtype=np.float32)
        reps = (np.prod(shape) + len(base) - 1) // len(base)
        x = np.tile(base, reps)[: np.prod(shape)].reshape(shape).astype(np.float32)

    weight = rng.uniform(0.8, 1.2, size=(HIDDEN,)).astype(np.float32)

    x_t = torch.from_numpy(x)
    w_t = torch.from_numpy(weight)
    ref = (x_t * torch.rsqrt(x_t.pow(2).mean(dim=-1, keepdim=True) + EPS) * w_t).numpy().astype(np.float32)
    return x, weight, ref


def _build_top(kernel_path: str):
    k = ExternalModule(top="rms_norm_720_fp32", impl_path=kernel_path, input_idx=[0, 2], output_idx=[1])

    @df.region()
    def top(A: float32[32], C: float32[32], P: float32[33]):
        @df.kernel(mapping=[1], args=[A, C, P])
        def core(a: float32[32] @ Ly, c: float32[32] @ Ly, p: float32[33] @ Ly):
            k(a, c, p)

    return top


def _test(kernel_path: str):
    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
        return

    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "basic").lower()
    repeat_count = int(os.environ.get("ALLO_REPEAT_COUNT", "1"))
    seeds = [0] if verify_mode == "basic" else [0, 7, 13, 23]
    cases = ["normal"] if verify_mode == "basic" else ["normal", "uniform_small", "uniform_large", "boundary_mix"]

    seq_sample_raw = os.environ.get("ALLO_SEQ_SAMPLE", "0")
    seq_indices = sorted(
        {
            idx
            for idx in (int(v.strip()) for v in seq_sample_raw.split(",") if v.strip())
            if 0 <= idx < SEQ
        }
    )
    if not seq_indices:
        seq_indices = [0]

    mod = df.build(_build_top(kernel_path), target="aie", profile=False, warmup=0, num_iters=1, project=TOP_PRJ_ABS_DIR)

    fails = []
    total = 0
    aggregate_max_abs = 0.0
    aggregate_mean_abs = 0.0
    aggregate_count = 0
    for seed in seeds:
        for case_name in cases:
            x, w, ref = _build_case(seed, case_name)
            out = np.zeros_like(x, dtype=np.float32)
            for n in range(BATCH):
                for s in seq_indices:
                    row = x[n, s, :]
                    inv_std = float(1.0 / np.sqrt(np.mean(row * row) + EPS))
                    for i0 in range(0, HIDDEN, CHUNK):
                        valid = min(CHUNK, HIDDEN - i0)
                        a = np.zeros((CHUNK,), dtype=np.float32)
                        a[:valid] = row[i0:i0 + valid]
                        p = np.zeros((CHUNK + 1,), dtype=np.float32)
                        p[:valid] = w[i0:i0 + valid]
                        p[CHUNK] = np.float32(inv_std)
                        o = np.zeros((CHUNK,), dtype=np.float32)
                        mod(a, o, p)
                        out[n, s, i0:i0 + valid] = o[:valid]
            total += 1
            out_eval = out[:, seq_indices, :]
            ref_eval = ref[:, seq_indices, :]
            if not np.isfinite(out_eval).all():
                fails.append(f"seed={seed}, case={case_name}: output contains non-finite values")
                continue
            try:
                np.testing.assert_allclose(out_eval, ref_eval, rtol=RTOL, atol=ATOL)
            except AssertionError as e:
                fails.append(f"seed={seed}, case={case_name}: {e}")
                continue

            diff = np.abs(out_eval - ref_eval)
            max_abs = float(np.max(diff))
            mean_abs = float(np.mean(diff))
            aggregate_max_abs = max(aggregate_max_abs, max_abs)
            aggregate_mean_abs += mean_abs
            aggregate_count += 1

            if seed == seeds[0] and case_name == cases[0] and repeat_count > 1:
                for rep in range(repeat_count - 1):
                    out_repeat = np.zeros_like(x, dtype=np.float32)
                    for n in range(BATCH):
                        for s in seq_indices:
                            row = x[n, s, :]
                            inv_std = float(1.0 / np.sqrt(np.mean(row * row) + EPS))
                            for i0 in range(0, HIDDEN, CHUNK):
                                valid = min(CHUNK, HIDDEN - i0)
                                a = np.zeros((CHUNK,), dtype=np.float32)
                                a[:valid] = row[i0:i0 + valid]
                                p = np.zeros((CHUNK + 1,), dtype=np.float32)
                                p[:valid] = w[i0:i0 + valid]
                                p[CHUNK] = np.float32(inv_std)
                                o = np.zeros((CHUNK,), dtype=np.float32)
                                mod(a, o, p)
                                out_repeat[n, s, i0:i0 + valid] = o[:valid]
                    out_repeat_eval = out_repeat[:, seq_indices, :]
                    if not np.allclose(out_repeat_eval, out_eval, rtol=0.0, atol=0.0):
                        fails.append(
                            f"seed={seed}, case={case_name}: repeated run #{rep + 2} produced different outputs"
                        )
                        break
                    try:
                        np.testing.assert_allclose(out_repeat_eval, ref_eval, rtol=RTOL, atol=ATOL)
                    except AssertionError as e:
                        fails.append(
                            f"seed={seed}, case={case_name}: repeated run #{rep + 2} mismatch: {e}"
                        )
                        break

            print(f"[CASE PASS] seed={seed}, case={case_name}, max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

    if aggregate_count > 0:
        print(
            f"Aggregate error stats: max_abs={aggregate_max_abs:.3e}, "
            f"mean_abs={aggregate_mean_abs / aggregate_count:.3e} over {aggregate_count} cases"
        )

    if fails:
        print("FAIL!")
        print(f"Verification failed: {len(fails)}/{total} cases failed")
        for msg in fails[:8]:
            print(f"  - {msg}")
    else:
        print(f"All verification cases passed: {total}/{total}")
        print("PASS!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="rms_norm_720_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test(args.kernel_path)
