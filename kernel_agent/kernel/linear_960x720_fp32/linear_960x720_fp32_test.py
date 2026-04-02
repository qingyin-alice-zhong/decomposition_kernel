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


KERNEL_NAME = "linear_960x720_fp32"
FULL_BATCH = 1
FULL_SEQ = 50
IN_FEATURES = 960
OUT_FEATURES = 720
HAS_BIAS = False

KERNEL_IN_FEATURES = 32
OUT_TILE = 1
ABI_TILE = 1
INPUT_SIZE = KERNEL_IN_FEATURES
OUTPUT_SIZE = ABI_TILE
PARAM_SIZE = KERNEL_IN_FEATURES + (1 if HAS_BIAS else 0)

NUMERIC_RTOL = 1e-4
NUMERIC_ATOL = 1e-4
Ly = Layout("R")


def build_case(seed: int, case_name: str):
    rng = np.random.default_rng(seed)
    if case_name == "normal":
        x = rng.standard_normal((FULL_BATCH, FULL_SEQ, IN_FEATURES), dtype=np.float32)
    elif case_name == "uniform_small":
        x = rng.uniform(-0.5, 0.5, size=(FULL_BATCH, FULL_SEQ, IN_FEATURES)).astype(np.float32)
    else:
        x = rng.uniform(-5.0, 5.0, size=(FULL_BATCH, FULL_SEQ, IN_FEATURES)).astype(np.float32)

    weight = (rng.standard_normal((OUT_FEATURES, IN_FEATURES), dtype=np.float32) * 0.05).astype(np.float32)
    bias = np.zeros((OUT_FEATURES,), dtype=np.float32)

    with torch.no_grad():
        ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(weight), None).cpu().numpy().astype(np.float32)
    return x, weight, bias, ref


def _build_top(kernel_path: str):
    linear_kernel = ExternalModule(top=KERNEL_NAME, impl_path=kernel_path, input_idx=[0, 2], output_idx=[1])

    @df.region()
    def top(A: float32[INPUT_SIZE], C: float32[OUTPUT_SIZE], P: float32[PARAM_SIZE]):
        @df.kernel(mapping=[1], args=[A, C, P])
        def core(local_A: float32[INPUT_SIZE] @ Ly, local_C: float32[OUTPUT_SIZE] @ Ly, local_P: float32[PARAM_SIZE] @ Ly):
            linear_kernel(local_A, local_C, local_P)

    return top


def run_tiled_linear(mod, x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    output = np.zeros((FULL_BATCH, FULL_SEQ, OUT_FEATURES), dtype=np.float32)
    seq_sample_raw = os.environ.get("ALLO_SEQ_SAMPLE", "0")
    seq_indices = sorted({idx for idx in (int(v.strip()) for v in seq_sample_raw.split(",") if v.strip()) if 0 <= idx < FULL_SEQ})
    if not seq_indices:
        seq_indices = [0]

    tile_count = 0
    for n in range(FULL_BATCH):
        for s in seq_indices:
            in_vec = np.ascontiguousarray(x[n, s, :], dtype=np.float32)
            for out_start in range(0, OUT_FEATURES, OUT_TILE):
                partial_sum = 0.0
                for in_start in range(0, IN_FEATURES, KERNEL_IN_FEATURES):
                    valid = min(KERNEL_IN_FEATURES, IN_FEATURES - in_start)
                    in_chunk = np.zeros((KERNEL_IN_FEATURES,), dtype=np.float32)
                    in_chunk[:valid] = in_vec[in_start:in_start + valid]

                    param = np.zeros((PARAM_SIZE,), dtype=np.float32)
                    param[:valid] = weight[out_start, in_start:in_start + valid]

                    out_vec = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
                    mod(in_chunk, out_vec, param)
                    partial_sum += out_vec[0]
                    tile_count += 1

                output[n, s, out_start] = partial_sum

    print(f"Executed {tile_count} tiles for {KERNEL_NAME} (sampled_seq={seq_indices})")
    print(f"Boundary check: seq_first={0 in seq_indices}, seq_last={FULL_SEQ - 1 in seq_indices}, sampled_seq={seq_indices}")
    return output, seq_indices


def _test(kernel_path: str):
    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "basic").lower()
    repeat_count = int(os.environ.get("ALLO_REPEAT_COUNT", "1"))
    strict_heavy = os.environ.get("ALLO_STRICT_HEAVY", "0") == "1"

    if verify_mode == "basic":
        seeds, cases = [0], ["normal"]
    elif strict_heavy:
        seeds, cases = [0, 7, 13, 23], ["normal", "uniform_small", "uniform_large"]
    else:
        seeds, cases = [0, 7], ["normal", "uniform_small"]

    top = _build_top(kernel_path)
    mod = df.build(top, target="aie", profile=False, warmup=0, num_iters=1, project=TOP_PRJ_ABS_DIR)

    failures = []
    total_cases = 0
    aggregate_count = 0
    for seed in seeds:
        for case_name in cases:
            x, weight, bias, ref = build_case(seed, case_name)
            out, seq_indices = run_tiled_linear(mod, x, weight, bias)
            total_cases += 1
            out_eval = out[:, seq_indices, :]
            ref_eval = ref[:, seq_indices, :]
            try:
                np.testing.assert_allclose(out_eval, ref_eval, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
            except AssertionError as e:
                failures.append(f"seed={seed}, case={case_name}: {e}")
                continue
            aggregate_count += 1
            print(f"[CASE PASS] seed={seed}, case={case_name}")

            if seed == seeds[0] and case_name == cases[0] and repeat_count > 1:
                for _ in range(repeat_count - 1):
                    out2, _ = run_tiled_linear(mod, x, weight, bias)
                    np.testing.assert_allclose(out2[:, seq_indices, :], ref_eval, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)

    if failures:
        print("FAIL!")
        for msg in failures[:8]:
            print(f"  - {msg}")
    elif total_cases == 0 or aggregate_count != total_cases:
        print("FAIL!")
        print(f"Verification bookkeeping mismatch: total_cases={total_cases}, aggregate_count={aggregate_count}")
    else:
        print(f"All verification cases passed: {total_cases}/{total_cases}")
        print("PASS!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="linear_960x720_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test(args.kernel_path)
