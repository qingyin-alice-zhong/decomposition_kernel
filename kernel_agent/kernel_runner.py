import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent


TIME_PATTERN = re.compile(r"Avg NPU execution time:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)")
MIN_TIME_PATTERN = re.compile(r"Min NPU execution time:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)")
NPU_TIME_SAMPLE_PATTERN = re.compile(r"NPU execution time:\s*([0-9]+(?:\.[0-9]+)?)\s*us")
CYCLE_PATTERN = re.compile(
    r"First/Min/Avg/Max cycles is\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*/\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*/\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*/\s*"
    r"([0-9]+(?:\.[0-9]+)?)"
)
TRACE_FAILURE_PATTERN = re.compile(r"Trace analysis failed:.*")


@dataclass
class KernelRunResult:
    kernel_name: str
    kernel_path: str
    test_path: str
    output_dir: str
    working_directory: str
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    compilation_success: bool
    execution_success: bool
    verification_success: bool
    skipped: bool
    vector_time_us: Optional[float]
    min_time_us: Optional[float]
    vector_time_cycles: Optional[float]
    min_cycles: Optional[float]
    avg_cycles: Optional[float]
    max_cycles: Optional[float]
    trace_analysis_message: Optional[str]

    def to_summary(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "kernel_path": self.kernel_path,
            "test_path": self.test_path,
            "output_dir": self.output_dir,
            "status": {
                "compilation_success": self.compilation_success,
                "execution_success": self.execution_success,
                "verification_success": self.verification_success,
                "skipped": self.skipped,
            },
            "performance_analysis": {
                "vector_time_us": self.vector_time_us,
                "min_time_us": self.min_time_us,
                "vector_time_cycles": self.vector_time_cycles,
                "min_cycles": self.min_cycles,
                "avg_cycles": self.avg_cycles,
                "max_cycles": self.max_cycles,
                "trace_analysis_message": self.trace_analysis_message,
            },
            "command_details": {
                "command": self.command,
                "working_directory": self.working_directory,
                "return_code": self.return_code,
            },
            "raw_output_lengths": {
                "stdout_length": len(self.stdout),
                "stderr_length": len(self.stderr),
            },
        }


def _convert_to_us(value: float, unit: str) -> float:
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    return value


def _extract_metrics(stdout: str) -> dict:
    metrics = {
        "verification_success": False,
        "skipped": False,
        "vector_time_us": None,
        "min_time_us": None,
        "vector_time_cycles": None,
        "min_cycles": None,
        "avg_cycles": None,
        "max_cycles": None,
        "trace_analysis_message": None,
    }

    if "PASS!" in stdout:
        metrics["verification_success"] = True
    elif "FAIL!" in stdout:
        metrics["verification_success"] = False

    if "Skipping AIE backend test" in stdout or "MLIR_AIE_INSTALL_DIR unset" in stdout:
        metrics["skipped"] = True

    time_match = TIME_PATTERN.search(stdout)
    if time_match:
        value = float(time_match.group(1))
        unit = time_match.group(2)
        metrics["vector_time_us"] = _convert_to_us(value, unit)

    min_time_match = MIN_TIME_PATTERN.search(stdout)
    if min_time_match:
        value = float(min_time_match.group(1))
        unit = min_time_match.group(2)
        metrics["min_time_us"] = _convert_to_us(value, unit)

    time_samples_us = [float(match) for match in NPU_TIME_SAMPLE_PATTERN.findall(stdout)]
    if time_samples_us:
        if metrics["vector_time_us"] is None:
            metrics["vector_time_us"] = sum(time_samples_us) / len(time_samples_us)
        if metrics["min_time_us"] is None:
            metrics["min_time_us"] = min(time_samples_us)

    cycle_match = CYCLE_PATTERN.search(stdout)
    if cycle_match:
        metrics["min_cycles"] = float(cycle_match.group(2))
        metrics["avg_cycles"] = float(cycle_match.group(3))
        metrics["max_cycles"] = float(cycle_match.group(4))
        metrics["vector_time_cycles"] = metrics["avg_cycles"]

    trace_failure_match = TRACE_FAILURE_PATTERN.search(stdout)
    if trace_failure_match:
        metrics["trace_analysis_message"] = trace_failure_match.group(0)
    elif "Wrote trace JSON:" in stdout:
        metrics["trace_analysis_message"] = "Trace analysis completed."

    return metrics


def _write_combined_output(output_path: Path, command: list[str], working_directory: Path, return_code: int, stdout: str, stderr: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(command))
        f.write("\n\nWORKING DIRECTORY:\n")
        f.write(str(working_directory))
        f.write("\n\nRETURN CODE:\n")
        f.write(str(return_code))
        f.write("\n\nSTDOUT:\n")
        f.write(stdout)
        f.write("\n\nSTDERR:\n")
        f.write(stderr)


def _normalize_process_output(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_kernel_test(kernel_path: Path, test_path: Path, output_dir: Path, timeout: int = 1800) -> KernelRunResult:
    kernel_name = kernel_path.stem
    working_directory = test_path.parent.resolve()
    command = [sys.executable, str(test_path.resolve()), "--kernel_path", str(kernel_path.resolve())]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(PROJECT_ROOT)
    )

    return_code = -1
    stdout = ""
    stderr = ""

    try:
        process = subprocess.run(
            command,
            cwd=working_directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return_code = process.returncode
        stdout = _normalize_process_output(process.stdout)
        stderr = _normalize_process_output(process.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_process_output(exc.stdout)
        stderr = _normalize_process_output(exc.stderr) + f"\nTimed out after {timeout} seconds."
    metrics = _extract_metrics(stdout)

    compilation_success = return_code == 0 and not metrics["skipped"]
    execution_success = return_code == 0 and not metrics["skipped"]

    compile_output_path = output_dir / f"{kernel_name}_compile_output.txt"
    execution_output_path = output_dir / f"{kernel_name}_execution_output.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_combined_output(
        compile_output_path,
        command,
        working_directory,
        return_code,
        stdout,
        stderr,
    )
    _write_combined_output(
        execution_output_path,
        command,
        working_directory,
        return_code,
        stdout,
        stderr,
    )

    return KernelRunResult(
        kernel_name=kernel_name,
        kernel_path=str(kernel_path.resolve()),
        test_path=str(test_path.resolve()),
        output_dir=str(output_dir.resolve()),
        working_directory=str(working_directory),
        command=command,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        compilation_success=compilation_success,
        execution_success=execution_success,
        verification_success=metrics["verification_success"],
        skipped=metrics["skipped"],
        vector_time_us=metrics["vector_time_us"],
        min_time_us=metrics["min_time_us"],
        vector_time_cycles=metrics["vector_time_cycles"],
        min_cycles=metrics["min_cycles"],
        avg_cycles=metrics["avg_cycles"],
        max_cycles=metrics["max_cycles"],
        trace_analysis_message=metrics["trace_analysis_message"],
    )
