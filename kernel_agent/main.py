import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from kernel_runner import run_kernel_test


def setup_logging(output_dir: Path) -> Path:
    """Set up logging to both console and a file inside the output directory."""
    log_file = output_dir / "run.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an external NPU kernel C/C++ file together with its test.py, "
            "capture compile/execute logs, and extract vector_time information."
        )
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        metavar="KERNEL",
        help=(
            "Kernel name stored under kernel/<name>/, for example 'add_offset_int8'. "
            "This resolves to kernel/add_offset_int8/add_offset_int8.cc and "
            "kernel/add_offset_int8/add_offset_int8_test.py automatically."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for executing the test.py script (default: 1800).",
    )
    return parser.parse_args()


def normalize_kernel_name(kernel_arg: str) -> str:
    kernel_name = Path(kernel_arg).stem
    if kernel_name.endswith("_test"):
        kernel_name = kernel_name[:-5]
    return kernel_name


def resolve_kernel_inputs(project_root: Path, kernel_arg: str) -> tuple[Path, Path]:
    kernel_name = normalize_kernel_name(kernel_arg)
    kernel_dir = project_root / "kernel" / kernel_name
    kernel_path = (kernel_dir / f"{kernel_name}.cc").resolve()
    test_path = (kernel_dir / f"{kernel_name}_test.py").resolve()
    return kernel_path, test_path


def validate_input_paths(kernel_path: Path, test_path: Path):
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
    if not kernel_path.is_file():
        raise ValueError(f"Kernel path is not a file: {kernel_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not test_path.is_file():
        raise ValueError(f"Test path is not a file: {test_path}")
    if test_path.suffix != ".py":
        raise ValueError(f"Test file must be a Python file: {test_path}")


def prepare_output_dir(project_root: Path, kernel_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "outputs" / f"{timestamp}_{kernel_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def copy_inputs_for_record(kernel_path: Path, test_path: Path, output_dir: Path):
    shutil.copy2(kernel_path, output_dir / kernel_path.name)
    shutil.copy2(test_path, output_dir / test_path.name)


def log_result_summary(summary: dict, output_dir: Path):
    status = summary["status"]
    performance = summary["performance_analysis"]

    logging.info("RESULTS SUMMARY")
    logging.info("=" * 20)
    logging.info(f"Kernel: {summary['kernel_name']}")
    logging.info(f"Kernel path: {summary['kernel_path']}")
    logging.info(f"Test path: {summary['test_path']}")
    logging.info(
        f"Compilation Success: {'✓' if status['compilation_success'] else '✗'}"
    )
    logging.info(
        f"Execution Success: {'✓' if status['execution_success'] else '✗'}"
    )
    logging.info(
        f"Verification Success: {'✓' if status['verification_success'] else '✗'}"
    )

    if performance["vector_time_us"] is not None:
        logging.info(f"Vector time: {performance['vector_time_us']:.6f} us")
    if performance["vector_time_cycles"] is not None:
        logging.info(f"Vector time: {performance['vector_time_cycles']:.6f} cycles")
    if performance["min_time_us"] is not None:
        logging.info(f"Min NPU time: {performance['min_time_us']:.6f} us")
    if performance["avg_cycles"] is not None:
        logging.info(f"Avg cycles: {performance['avg_cycles']:.6f}")

    if performance["trace_analysis_message"]:
        logging.info(f"Trace analysis: {performance['trace_analysis_message']}")

    logging.info(f"Artifacts saved in: {output_dir}")
    logging.info("=" * 60)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    try:
        kernel_path, test_path = resolve_kernel_inputs(project_root, args.kernel)
        validate_input_paths(kernel_path, test_path)

        output_dir = prepare_output_dir(project_root, kernel_path)
        log_file = setup_logging(output_dir)
        logging.info(f"Logging initialized. Log file: {log_file}")
        logging.info(f"Project root: {project_root}")
        logging.info(f"Kernel file: {kernel_path}")
        logging.info(f"Test file: {test_path}")

        copy_inputs_for_record(kernel_path, test_path, output_dir)

        result = run_kernel_test(
            kernel_path=kernel_path,
            test_path=test_path,
            output_dir=output_dir,
            timeout=args.timeout,
        )

        summary = result.to_summary()
        summary_path = output_dir / f"{kernel_path.stem}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Summary saved to {summary_path}")
        log_result_summary(summary, output_dir)
    except Exception as exc:
        logging.error(f"Kernel run failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
