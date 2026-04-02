import os
from dataclasses import dataclass
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


@dataclass
class LinearTiledConfig:
    kernel_name: str
    full_batch: int
    full_seq: int
    in_features: int
    out_features: int
    has_bias: bool
    kernel_path: str
    rtol: float = 1e-4
    atol: float = 1e-4


def run_linear_tiled_test(cfg: LinearTiledConfig):
    input_size = 32
    output_size = 1
    param_size = 33 if cfg.has_bias else 32

    def build_case(seed: int, case_name: str):
        rng = np.random.default_rng(seed)
        if case_name == "normal":
            x = rng.standard_normal((cfg.full_batch, cfg.full_seq, cfg.in_features), dtype=np.float32)
        elif case_name == "uniform_small":
            x = rng.uniform(-0.5, 0.5, size=(cfg.full_batch, cfg.full_seq, cfg.in_features)).astype(np.float32)
        else:
            x = rng.uniform(-5.0, 5.0, size=(cfg.full_batch, cfg.full_seq, cfg.in_features)).astype(np.float32)

        weight = (rng.standard_normal((cfg.out_features, cfg.in_features), dtype=np.float32) * 0.05).astype(np.float32)
        bias = (rng.standard_normal((cfg.out_features,), dtype=np.float32) * 0.02).astype(np.float32) if cfg.has_bias else np.zeros((cfg.out_features,), dtype=np.float32)

        with torch.no_grad():
            bias_tensor = torch.from_numpy(bias) if cfg.has_bias else None
            ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(weight), bias_tensor).cpu().numpy().astype(np.float32)
        return x, weight, bias, ref

    linear_kernel = ExternalModule(top=cfg.kernel_name, impl_path=cfg.kernel_path, input_idx=[0, 2], output_idx=[1])

    @df.region()
    def top(A: float32[input_size], C: float32[output_size], P: float32[param_size]):
        @df.kernel(mapping=[1], args=[A, C, P])
        def core(local_A: float32[input_size] @ Ly, local_C: float32[output_size] @ Ly, local_P: float32[param_size] @ Ly):
            linear_kernel(local_A, local_C, local_P)

    mod = df.build(top, target="aie", profile=False, warmup=0, num_iters=1, project=TOP_PRJ_ABS_DIR)

    def run_once(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
        out = np.zeros((cfg.full_batch, cfg.full_seq, cfg.out_features), dtype=np.float32)
        seq_sample_raw = os.environ.get("ALLO_SEQ_SAMPLE", "0")
        seq_indices = sorted({idx for idx in (int(v.strip()) for v in seq_sample_raw.split(",") if v.strip()) if 0 <= idx < cfg.full_seq})
        if not seq_indices:
            seq_indices = [0]

        tile_count = 0
        for n in range(cfg.full_batch):
            for s in seq_indices:
                in_vec = np.ascontiguousarray(x[n, s, :], dtype=np.float32)
                for o in range(cfg.out_features):
                    psum = 0.0
                    for i0 in range(0, cfg.in_features, 32):
                        valid = min(32, cfg.in_features - i0)
                        in_chunk = np.zeros((32,), dtype=np.float32)
                        in_chunk[:valid] = in_vec[i0:i0 + valid]
                        param = np.zeros((param_size,), dtype=np.float32)
                        param[:valid] = weight[o, i0:i0 + valid]
                        if cfg.has_bias and i0 == 0:
                            param[32] = bias[o]
                        out_vec = np.zeros((1,), dtype=np.float32)
                        mod(in_chunk, out_vec, param)
                        psum += out_vec[0]
                        tile_count += 1
                    out[n, s, o] = psum

        print(f"Executed {tile_count} tiles for {cfg.kernel_name} (sampled_seq={seq_indices})")
        print(f"Boundary check: seq_first={0 in seq_indices}, seq_last={cfg.full_seq - 1 in seq_indices}, sampled_seq={seq_indices}")
        return out, seq_indices

    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "basic").lower()
    repeat_count = int(os.environ.get("ALLO_REPEAT_COUNT", "1"))
    strict_heavy = os.environ.get("ALLO_STRICT_HEAVY", "0") == "1"

    if verify_mode == "basic":
        seeds, cases = [0], ["normal"]
    elif strict_heavy:
        seeds, cases = [0, 7, 13, 23], ["normal", "uniform_small", "uniform_large"]
    else:
        seeds, cases = [0, 7], ["normal", "uniform_small"]

    failures = []
    total_cases = 0
    aggregate_count = 0

    for seed in seeds:
        for case_name in cases:
            x, weight, bias, ref = build_case(seed, case_name)
            out, seq_indices = run_once(x, weight, bias)
            total_cases += 1
            out_eval = out[:, seq_indices, :]
            ref_eval = ref[:, seq_indices, :]
            try:
                np.testing.assert_allclose(out_eval, ref_eval, rtol=cfg.rtol, atol=cfg.atol)
            except AssertionError as e:
                failures.append(f"seed={seed}, case={case_name}: {e}")
                continue

            aggregate_count += 1
            print(f"[CASE PASS] seed={seed}, case={case_name}")

            if seed == seeds[0] and case_name == cases[0] and repeat_count > 1:
                for _ in range(repeat_count - 1):
                    out2, _ = run_once(x, weight, bias)
                    np.testing.assert_allclose(out2[:, seq_indices, :], ref_eval, rtol=cfg.rtol, atol=cfg.atol)

    if failures:
        print("FAIL!")
        print(f"Verification failed: {len(failures)}/{total_cases} cases failed")
        for msg in failures[:8]:
            print(f"  - {msg}")
    elif total_cases == 0 or aggregate_count != total_cases:
        print("FAIL!")
        print(f"Verification bookkeeping mismatch: total_cases={total_cases}, aggregate_count={aggregate_count}")
    else:
        print(f"All verification cases passed: {total_cases}/{total_cases}")
        print("PASS!")
