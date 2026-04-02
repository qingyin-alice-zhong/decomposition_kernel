import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch

import allo.dataflow as df
from allo.ir.types import float32, int32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import TOP_PRJ_ABS_DIR


FULL_SEQ = 1024
FULL_HIDDEN = 768
VOCAB_SIZE = 1024

TILE_SEQ = 16
TILE_HIDDEN = 64

INPUT_SIZE = TILE_SEQ
OUTPUT_SIZE = TILE_SEQ * TILE_HIDDEN
TABLE_SIZE = TILE_SEQ * TILE_HIDDEN
META_SIZE = 8
PARAM_SIZE = TABLE_SIZE + META_SIZE

Ly = Layout("R")


def build_weight(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((VOCAB_SIZE, FULL_HIDDEN), dtype=np.float32) * 0.05).astype(np.float32)
    return weight


def build_reference(weight: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        ref = torch.from_numpy(weight)[torch.from_numpy(position_ids).to(torch.int64)].cpu().numpy().astype(np.float32)
    return ref


def build_position_case(case_name: str, seed: int = 0) -> np.ndarray:
    if case_name == "sequential":
        ids = np.arange(FULL_SEQ, dtype=np.int32)
    elif case_name == "reverse":
        ids = np.arange(FULL_SEQ - 1, -1, -1, dtype=np.int32)
    elif case_name == "random":
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, VOCAB_SIZE, size=(FULL_SEQ,), dtype=np.int32)
    elif case_name == "stride_wrap":
        ids = ((np.arange(FULL_SEQ, dtype=np.int32) * 37) % VOCAB_SIZE).astype(np.int32)
    elif case_name == "boundary_mix":
        base = np.array([0, VOCAB_SIZE - 1, 1, VOCAB_SIZE - 2], dtype=np.int32)
        reps = (FULL_SEQ + len(base) - 1) // len(base)
        ids = np.tile(base, reps)[:FULL_SEQ]
    else:
        raise ValueError(f"Unknown position_ids case: {case_name}")
    return ids.reshape(1, FULL_SEQ)


def _build_top(kernel_path: str):
    embedding_kernel = ExternalModule(
        top="embedding_1024x768_fp32",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    @df.region()
    def top(A: int32[INPUT_SIZE], C: float32[OUTPUT_SIZE], P: float32[PARAM_SIZE]):
        @df.kernel(mapping=[1], args=[A, C, P])
        def core(local_A: int32[INPUT_SIZE] @ Ly, local_C: float32[OUTPUT_SIZE] @ Ly, local_P: float32[PARAM_SIZE] @ Ly):
            embedding_kernel(local_A, local_C, local_P)

    return top


def run_tiled_embedding(mod, position_ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    output = np.zeros((1, FULL_SEQ, FULL_HIDDEN), dtype=np.float32)

    warm_ids = np.zeros((INPUT_SIZE,), dtype=np.int32)
    warm_out = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
    warm_param = np.zeros((PARAM_SIZE,), dtype=np.float32)
    mod(warm_ids, warm_out, warm_param)

    tile_count = 0
    for hidden_start in range(0, FULL_HIDDEN, TILE_HIDDEN):
        for seq_start in range(0, FULL_SEQ, TILE_SEQ):
            ids_tile = position_ids[0, seq_start:seq_start + TILE_SEQ].astype(np.int32, copy=False)
            table = weight[ids_tile, hidden_start:hidden_start + TILE_HIDDEN]

            param = np.zeros((PARAM_SIZE,), dtype=np.float32)
            param[:TABLE_SIZE] = table.reshape(-1)

            out_tile = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
            mod(ids_tile, out_tile, param)

            output[0, seq_start:seq_start + TILE_SEQ, hidden_start:hidden_start + TILE_HIDDEN] = out_tile.reshape(TILE_SEQ, TILE_HIDDEN)
            tile_count += 1

    print(f"Executed {tile_count} tiles for embedding_1024x768_fp32")
    return output


def _test_embedding_1024x768_fp32(kernel_path: str):
    print(
        "Running tiled validation for embedding_1024x768_fp32: "
        f"vocab={VOCAB_SIZE}, seq={FULL_SEQ}, hidden={FULL_HIDDEN}, "
        f"tile=(seq={TILE_SEQ}, hidden={TILE_HIDDEN})"
    )

    verify_mode = os.environ.get("ALLO_VERIFY_MODE", "full").lower()
    if verify_mode == "basic":
        seeds = [0]
        cases = ["sequential"]
    else:
        seeds = [0, 7]
        cases = ["sequential", "reverse", "random", "stride_wrap", "boundary_mix"]

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
        for seed in seeds:
            weight = build_weight(seed=seed)
            for case_name in cases:
                case_seed = seed + 123 if case_name == "random" else 0
                position_ids = build_position_case(case_name, seed=case_seed)
                ref = build_reference(weight, position_ids)

                output = run_tiled_embedding(mod, position_ids, weight)
                total_cases += 1
                try:
                    np.testing.assert_allclose(output, ref, rtol=1e-5, atol=1e-5)
                except AssertionError as e:
                    failures.append(f"seed={seed}, case={case_name}: {str(e)}")
                    continue

                if seed == seeds[0] and case_name == cases[0]:
                    output_repeat = run_tiled_embedding(mod, position_ids, weight)
                    if not np.allclose(output_repeat, output, rtol=0.0, atol=0.0):
                        failures.append(
                            f"seed={seed}, case={case_name}: repeated run produced different outputs"
                        )

                print(f"[CASE PASS] seed={seed}, case={case_name}")

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
    parser.add_argument("--kernel_path", type=str, default="embedding_1024x768_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_embedding_1024x768_fp32(args.kernel_path)
