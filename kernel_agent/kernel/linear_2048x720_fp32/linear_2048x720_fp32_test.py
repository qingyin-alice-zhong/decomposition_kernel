import argparse
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linear_tiled_micro_test_common import LinearTiledConfig, run_linear_tiled_test
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import TOP_PRJ_ABS_DIR


def _test(kernel_path: str):
    run_linear_tiled_test(
        LinearTiledConfig(
            kernel_name="linear_2048x720_fp32",
            full_batch=1,
            full_seq=50,
            in_features=2048,
            out_features=720,
            has_bias=False,
            kernel_path=kernel_path,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="linear_2048x720_fp32.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test(args.kernel_path)
