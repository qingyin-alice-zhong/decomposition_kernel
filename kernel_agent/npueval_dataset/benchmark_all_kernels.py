#!/usr/bin/env python3
"""
Benchmark all kernels in the npueval_dataset directory.
Runs each kernel's test.py and extracts performance metrics.
"""

import os
import json
import subprocess
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

def find_kernel_directories(dataset_path: str) -> List[str]:
    """Find all directories that contain test.py files (kernel directories)."""
    kernel_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item not in ['__pycache__', 'worked']:
            test_file = os.path.join(item_path, 'test.py')
            if os.path.exists(test_file):
                kernel_dirs.append(item)
    return sorted(kernel_dirs)

def extract_performance_metrics(output: str) -> Dict[str, Any]:
    """Extract performance metrics from test output."""
    metrics = {
        'avg_npu_time_us': None,
        'min_npu_time_us': None,
        'avg_cycles': None,
        'min_cycles': None,
        'max_cycles': None,
        'num_invocations': None,
        'status': 'unknown'
    }
    
    # Check if test passed or failed
    if 'PASS!' in output:
        metrics['status'] = 'pass'
    elif 'FAIL!' in output:
        metrics['status'] = 'fail'
    elif 'Traceback' in output or 'Error' in output:
        metrics['status'] = 'error'
    
    # Extract NPU execution times
    npu_avg_match = re.search(r'Avg NPU execution time:\s*([\d.]+)us', output)
    if npu_avg_match:
        metrics['avg_npu_time_us'] = float(npu_avg_match.group(1))
    
    npu_min_match = re.search(r'Min NPU execution time:\s*([\d.]+)us', output)
    if npu_min_match:
        metrics['min_npu_time_us'] = float(npu_min_match.group(1))
    
    # Extract cycle counts (from trace analysis)
    cycles_match = re.search(r'First/Min/Avg/Max cycles is\s*(\d+)/\s*(\d+)/\s*([\d.]+)/\s*(\d+)', output)
    if cycles_match:
        metrics['min_cycles'] = int(cycles_match.group(2))
        metrics['avg_cycles'] = float(cycles_match.group(3))
        metrics['max_cycles'] = int(cycles_match.group(4))
    
    # Extract number of invocations
    invocations_match = re.search(r'Total number of full kernel invocations is\s*(\d+)', output)
    if invocations_match:
        metrics['num_invocations'] = int(invocations_match.group(1))
    
    return metrics

def run_kernel_test(kernel_dir: str, dataset_path: str, timeout: int = 300) -> Dict[str, Any]:
    """Run a single kernel test and return results."""
    kernel_path = os.path.join(dataset_path, kernel_dir)
    
    result = {
        'kernel_name': kernel_dir,
        'success': False,
        'metrics': {},
        'error_message': None,
        'execution_time_seconds': None
    }
    
    try:
        print(f"Testing {kernel_dir}...", end=' ', flush=True)
        start_time = time.time()
        
        # Change to kernel directory and run test
        process = subprocess.run(
            ['python', 'test.py', '--kernel_path', 'canonical_scalar_allo.cc'],
            cwd=kernel_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        result['execution_time_seconds'] = execution_time
        
        # Combine stdout and stderr for analysis
        output = process.stdout + process.stderr
        
        # Extract metrics regardless of return code
        metrics = extract_performance_metrics(output)
        result['metrics'] = metrics
        
        if process.returncode == 0 and metrics['status'] == 'pass':
            result['success'] = True
            print(f"✅ PASS ({execution_time:.1f}s)")
            if metrics['avg_cycles']:
                print(f"   Avg cycles: {metrics['avg_cycles']:.1f}")
        else:
            result['success'] = False
            if metrics['status'] == 'fail':
                print(f"❌ FAIL ({execution_time:.1f}s)")
            elif metrics['status'] == 'error':
                print(f"💥 ERROR ({execution_time:.1f}s)")
            else:
                print(f"❓ UNKNOWN ({execution_time:.1f}s)")
            
            # Extract error message for debugging
            if 'Traceback' in output:
                # Get the last few lines of traceback
                lines = output.split('\n')
                traceback_start = -1
                for i, line in enumerate(lines):
                    if 'Traceback' in line:
                        traceback_start = i
                        break
                if traceback_start >= 0:
                    result['error_message'] = '\n'.join(lines[traceback_start:traceback_start+10])
            elif process.returncode != 0:
                result['error_message'] = f"Process exited with code {process.returncode}"
        
    except subprocess.TimeoutExpired:
        result['error_message'] = f"Test timed out after {timeout} seconds"
        print(f"⏰ TIMEOUT ({timeout}s)")
    except Exception as e:
        result['error_message'] = str(e)
        print(f"💥 EXCEPTION: {e}")
    
    return result

def main():
    dataset_path = "/home/jl4257/llm_codegen/npueval_dataset"
    output_file = os.path.join(dataset_path, "kernel_benchmark_results.json")
    
    print("🚀 Starting comprehensive kernel benchmarking...")
    print(f"Dataset path: {dataset_path}")
    print(f"Output file: {output_file}")
    print()
    
    # Find all kernel directories
    kernel_dirs = find_kernel_directories(dataset_path)
    print(f"Found {len(kernel_dirs)} kernel directories:")
    for i, kernel_dir in enumerate(kernel_dirs, 1):
        print(f"  {i:2d}. {kernel_dir}")
    print()
    
    # Run all tests
    all_results = []
    start_time = time.time()
    
    passed_count = 0
    failed_count = 0
    error_count = 0
    
    for i, kernel_dir in enumerate(kernel_dirs, 1):
        print(f"[{i:2d}/{len(kernel_dirs)}] ", end='')
        result = run_kernel_test(kernel_dir, dataset_path)
        all_results.append(result)
        
        if result['success']:
            passed_count += 1
        elif result['metrics'].get('status') == 'fail':
            failed_count += 1
        else:
            error_count += 1
    
    total_time = time.time() - start_time
    
    # Create summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_kernels': len(kernel_dirs),
        'passed': passed_count,
        'failed': failed_count,
        'errors': error_count,
        'total_execution_time_seconds': total_time,
        'success_rate': passed_count / len(kernel_dirs) if kernel_dirs else 0
    }
    
    # Compile final results
    benchmark_data = {
        'summary': summary,
        'results': all_results
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    # Print summary
    print()
    print("=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total kernels tested: {len(kernel_dirs)}")
    print(f"✅ Passed: {passed_count} ({passed_count/len(kernel_dirs)*100:.1f}%)")
    print(f"❌ Failed: {failed_count} ({failed_count/len(kernel_dirs)*100:.1f}%)")
    print(f"💥 Errors: {error_count} ({error_count/len(kernel_dirs)*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📁 Results saved to: {output_file}")
    
    # Show top performers by cycles
    passed_results = [r for r in all_results if r['success'] and r['metrics'].get('avg_cycles')]
    if passed_results:
        passed_results.sort(key=lambda x: x['metrics']['avg_cycles'])
        print()
        print("🏆 TOP 10 FASTEST KERNELS (by avg cycles):")
        for i, result in enumerate(passed_results[:10], 1):
            cycles = result['metrics']['avg_cycles']
            name = result['kernel_name']
            print(f"  {i:2d}. {name:<30} {cycles:>8.1f} cycles")
    
    print()
    print("✨ Benchmarking complete!")

if __name__ == "__main__":
    main()
