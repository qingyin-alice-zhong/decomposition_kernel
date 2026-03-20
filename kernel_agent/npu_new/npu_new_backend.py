import os
import re
import subprocess
from typing import Dict
from backend import Backend
from .npueval_config import (
    get_npueval_kernels,
    get_npueval_kernel_descriptions,
    get_npueval_kernel_paths,
    get_npueval_testbench_paths,
    get_npueval_scalar_times
)

class NPUNewBackend(Backend):
    """AMD Ryzen NPU new backend implementation: this uses Allo for the testbench. So the prompt and compile/run commands are different. """

    def __init__(self):
        super().__init__("npu_new")
        
        # self.api_docs_path = "/home/jl4257/.conda/envs/aie/lib/python3.12/site-packages/mlir_aie/include/aie_api/aie_doc.hpp"
        self.api_docs_path = "backends/npu_new/api_doc.md"
        
        # NPU-specific kernels (these would be different from CPU kernels)
        # Legacy kernels (commented out for reference)
        legacy_kernels = {
            # "scale": "unittest/kernels_func/scale.cc",
            # "mv": "unittest/kernels_func/mv.cc",
            # "gemm_kernel": "unittest/kernels_func/gemm_kernel.cc",
            # "conv": "unittest/kernels_func/conv.cc",
            # "softmax": "unittest/kernels_func/softmax.cc",
            # "silu": "unittest/kernels_func/silu.cc",
            # "masked_softmax": "unittest/kernels_func/masked_softmax.cc",
            # "gelu": "unittest/kernels_func/gelu.cc",
            # "conv_small": "unittest/kernels_func/conv_small.cc",
            # "attention": "unittest/kernels_func/attention.cc",
            # "reduce_add": "unittest/kernels_func/reduce_add.cc",
            # "reduce_max": "unittest/kernels_func/reduce_max.cc",
            # "reduce_min": "unittest/kernels_func/reduce_min.cc",
        }
        
        # NPUEval dataset kernels - automatically generated from configuration
        self.npu_kernel_path = get_npueval_kernel_paths()
        # NPUEval dataset kernel descriptions - automatically generated from configuration
        self.npu_kernel_description = get_npueval_kernel_descriptions()
        # NPUEval dataset testbench paths - automatically generated from configuration
        self.npu_kernel_testbench_path = get_npueval_testbench_paths()

        # NPUEval dataset scalar times - automatically generated from configuration
        self.npu_kernel_scalar_time = get_npueval_scalar_times()
        
        self.input_example_path = "backends/npu_new/add_kernel_func_sig.cc"
        self.output_example_path = "backends/npu_new/add_kernel_impl.cc"

        self.torch_ref_example_path = "backends/npu_new/add_torch_ref.py"
    
    def get_available_kernels(self) -> Dict[str, str]:
        """Get available NPU kernels."""
        return self.npu_kernel_path.copy()
    
    def create_prompt(self, input_code: str, kernel_name: str) -> str:
        """Create NPU-specific optimization prompt."""

        # Example code
        with open(self.torch_ref_example_path, "r") as f:
            torch_ref_example = f.read()

        with open(self.input_example_path, "r") as f:
            input_example = f.read()
        with open(self.output_example_path, "r") as f:
            output_example = f.read()
        
        # Extract the Pytorch reference code from the testbench code
        with open(self.npu_kernel_testbench_path[kernel_name], "r") as f:
            testbench_code = f.read()
        ref_code = re.search(r"# Reference code starts\n(.*?)# Reference code ends", testbench_code, re.DOTALL).group(1)


        api_docs = ""
        if os.path.exists(self.api_docs_path):
            with open(self.api_docs_path, "r") as f:
                api_docs = f.read()
        else:
            api_docs = "Documentation file not found."

        return f"""Act as an experienced performance kernel engineer focusing on AMD Ryzen AI NPUs powered by the AI Engine. Implement low-level kernel implementations in C++ using the AI Engine's API. You will be provided with the AI Engine API documentation. Your task is to finish the optimized kernel implementation given the function signature. You will be provided with the PyTorch/NumPy reference code for the kernel. 

# Steps
1. **Understand Hardware Specifications**: 
   - Familiarize yourself with the architecture and capabilities of AMD Ryzen AI NPUs.
   - Consult the latest documentation and resources provided by AMD regarding AI Engine specifications.
2. **AI Engine API Utilization**:
   - Explore the AI Engine's API for available functionalities and optimizations.
   - Identify key functions and methods within the API that are relevant to your kernel implementation.
3. **Kernel Implementation**:
   - Write efficient kernel code in C++ that leverages the AI Engine API optimally.
   - Ensure the implementation aligns with best practices in performance and low-level optimizations for NPUs.

# Output Format
Produce C++ code snippets as required for the kernel implementation. You will be provided with the function call in `extern "C"` blocks, which will call the internal function implementation. You should complete the function implementation in the C++ template format. Do not include any other text other than the entire code snippet. ONLY output the entire code snippet. The code should be complete with the necessary includes, function implementation, and extern "C" block. 
   
# AI Engine API Documentation
Here are some relevant parts of the AI Engine API documentation that might be useful:
```
{api_docs}
```

# Examples
## Input
Implement an add kernel that does the pointwise addition of 2 tensors. 
The PyTorch/NumPy reference code is as follows: 
```python
{torch_ref_example}
```

The kernel code with the function signature is as follows: 
```cpp
{input_example}
```
It includes the necessary #define, #include, and function signature wrapped in `extern "C"`. 
You should complete the function implementation with vectorization, while keeping the function signature and extern "C" block unchanged. Make sure the function matches the PyTorch/NumPy reference code. 

## Output
```cpp
{output_example}
```

# Task
Now implement a {self.npu_kernel_description[kernel_name]} kernel: {self.npu_kernel_description[kernel_name]}. 
Its reference code is as follows: 
```python
{ref_code}
```

The kernel code with the function signature is as follows: 
```cpp
{input_code}
```

You should complete the function implementation with vectorization, while keeping the function signature and extern "C" block unchanged and exactly the same as the input code (e.g., the exact number of dimensions of the input tensors should not be changed to constant expressions in your code). Make sure the function matches the PyTorch/NumPy reference code. Note that you should reserve the `event0()` and `event1()` calls in your code, as they are used to measure the execution time of the kernel.
Only output the entire code snippet in code block. Do not include any other text other than the entire code snippet. 
"""
    
    
    def create_output(self, input_complete_code: str, llm_gen_code: str, kernel_name: str) -> str:
        """Return the output complete code. 
        Here we directly return the code generated by the LLM, since we assume LLM will generate the complete code here. """
        return llm_gen_code
    
    def get_output_code_name(self, kernel_name: str) -> str:
        """Get the output code name."""
        return f"{kernel_name}.cc"
    
    def _filter_compile_warnings(self, compile_error: str) -> str:
        """Filter out compiler setup noise at the beginning and Python traceback at the end."""

        # separate `compile_error` into lines
        compile_error_lines = compile_error.split('\n')
        
        filtered = []
        skip_indices = set()
        n = len(compile_error_lines)
        # Remove everything before and including 'End of search list.'
        start_idx = 0
        for i, line in enumerate(compile_error_lines):
            if 'End of search list.' in line:
                start_idx = i + 1
                # increase the start_idx by 25 to skip the warning block (it always appears)
                start_idx += 25
                break

        # Remove everything from the first Traceback to the end
        end_idx = n
        for i in range(start_idx, n):
            if compile_error_lines[i].startswith('Traceback'):
                end_idx = i
                break
        # Now filter
        for i in range(start_idx, end_idx):
            if i in skip_indices:
                continue
            filtered.append(compile_error_lines[i])
        return '\n'.join(filtered)

    def compile_and_run(self, output_code_path: str, kernel_name: str, 
                       output_dir: str) -> Dict:
        """Compile and run C++ code on NPU using make with VPATH override."""
        # NPU compilation requires using the mlir-aie build system
        testbench_path = self.npu_kernel_testbench_path[kernel_name]
        
        compile_results = {
            "source_file": output_code_path,
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": ""
        }
        
        try:
            # Run the testbench with the kernel path (treating this as compilation step)
            compile_cmd = f"python {testbench_path} --kernel_path {os.path.abspath(output_code_path)}"

            compile_process = subprocess.run(
                compile_cmd,
                shell=True,  # Use shell to inherit full environment
                capture_output=True,
                text=True,
                timeout=120
            )
            
            compile_results["compile_cmd"] = compile_cmd
            compile_results["compile_output"] = compile_process.stdout
            # Filter out unwanted warnings from compile error
            raw_compile_error = compile_process.stderr
            # breakpoint()
            compile_results["compile_error"] = self._filter_compile_warnings(raw_compile_error)
            # breakpoint()
            compile_results["compilation_successful"] = compile_process.returncode == 0
            
            # If compilation successful, skip runtime execution
            if compile_results["compilation_successful"]:
                compile_results["execution_successful"] = True
                compile_results["run_output"] = compile_results["compile_output"] # since we skip the runtime execution, we just use the compile output as the run output
                compile_results["run_error"] = ""
                
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation timeout"
        except Exception as e:
            compile_results["compile_error"] = f"Error during compilation: {str(e)}"
        
        return compile_results
    
    def analyze_run_output(self, run_output: str, kernel_name: str) -> Dict:
        """Analyze NPU benchmark output."""
        analysis = {
            "verification_success": False,
            # "runtime": None, # unit in microseconds
            "scalar_time": None,
            "vector_time": None,
            "speedup": None,
        }
        
        lines = run_output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for verification results - simple PASS! or FAIL!
            if line == "PASS!":
                analysis["verification_success"] = True
            elif line == "FAIL!":
                analysis["verification_success"] = False
            
            # Look for NPU timing information
            # Format: "Avg NPU execution time: 115.689us"
            elif "Avg NPU execution time:" in line:
                try:
                    # Extract the time value and unit
                    time_part = line.split("Avg NPU execution time:")[-1].strip()
                    
                    if time_part.endswith("us"):
                        # Remove "us" and convert to float, keep in microseconds
                        time_str = time_part.replace("us", "").strip()
                        analysis["vector_time"] = float(time_str)
                    elif time_part.endswith("ms"):
                        # Remove "ms" and convert to microseconds
                        time_str = time_part.replace("ms", "").strip()
                        time_ms = float(time_str)
                        analysis["vector_time"] = time_ms * 1000.0  # Convert ms to us
                    elif time_part.endswith("s"):
                        # Remove "s" and convert to microseconds
                        time_str = time_part.replace("s", "").strip()
                        time_s = float(time_str)
                        analysis["vector_time"] = time_s * 1_000_000.0  # Convert s to us
                        
                except (ValueError, IndexError):
                    # If parsing fails, leave runtime as None
                    pass


            # Look for NPU cycle timing info
            # Format: "First/Min/Avg/Max cycles is 50/ 50/ 50.0/ 50"
            ## take avg cycles as the analysis["vector_time"]
            elif "First/Min/Avg/Max cycles is" in line:
                try:
                    # Extract Avg cycles (3rd value) robustly
                    match = re.search(r'First/Min/Avg/Max cycles is\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)', line)
                    if match:
                        analysis["vector_time"] = float(match.group(3))
                    else:
                        parts = [p.strip() for p in line.split("First/Min/Avg/Max cycles is")[-1].split('/')]
                        if len(parts) >= 3:
                            analysis["vector_time"] = float(parts[2])
                except (ValueError, IndexError):
                    pass

            
        
        if analysis["vector_time"] is not None:
            analysis["scalar_time"] = self.npu_kernel_scalar_time[kernel_name]
            analysis["speedup"] = analysis["scalar_time"] / analysis["vector_time"]

        return analysis
    
    def get_system_info(self) -> str:
        """Get NPU system information."""
        return """NPU Architecture:
- AMD Ryzen AI NPU (XDNA architecture)
- Dedicated AI acceleration units
- Optimized for INT8/FP16 computations
- Specialized memory subsystem for AI workloads
- Support for quantized operations""" 
    
    def _extract_wrapped_func_block(self, complete_code: str) -> str:
        """Extract the extern 'C' function block from C++ code."""
        lines = complete_code.split('\n')
        
        # Find the start of extern "C" block
        extern_start_line = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('extern "C"') and '{' in stripped:
                extern_start_line = i
                break
            elif stripped == 'extern "C"':
                # Check if next line has opening brace
                if i + 1 < len(lines) and '{' in lines[i + 1]:
                    extern_start_line = i
                    break
        
        if extern_start_line == -1:
            return "// No extern C block found"
        
        # Find the matching closing brace
        open_braces = 0
        extern_end_line = -1
        found_opening_brace = False
        
        for i in range(extern_start_line, len(lines)):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    open_braces += 1
                    found_opening_brace = True
                elif char == '}':
                    open_braces -= 1
            
            # If we found the opening brace and braces are balanced, we found the end
            if found_opening_brace and open_braces == 0:
                extern_end_line = i
                break
        
        if extern_end_line == -1:
            return "// Could not find end of extern C block"
        
        # Extract the extern "C" block
        extern_block_lines = lines[extern_start_line:extern_end_line + 1]
        return '\n'.join(extern_block_lines)

    @property
    def show_speedup_in_feedback(self):
        return True