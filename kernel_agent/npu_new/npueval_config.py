"""
NPUEval dataset configuration for NPU backend.

This module defines the NPUEval kernels in a clean, maintainable way.
Adding a new kernel only requires adding it to the NPUEVAL_KERNELS list.
"""

# NPUEval kernel definitions (alphabetically ordered)
# Each entry contains: (kernel_name, description, scalar_time_cycles)
NPUEVAL_KERNELS = [
    ("abs_int8", "This AIE kernel implements an abs function on an input vector.", 9990),
    ("add_offset_int8", "This AIE kernel adds a scalar int8 offset to every element of the input int8_t vector (length 256), and writes the result to the output buffer.", 2566),
    ("argmax_bfloat16", "This AIE kernel calculates the argmax of a given bfloat16 vector, returning a single int32 value for the maximum argument.", 2472),
    ("argmax_int32", "This AIE kernel calculates the argmax (the index of the largest element) of a given int32 vector of length 256 and stores the result in a single int32 element output buffer.", 1792),
    ("argmin_bfloat16", "This AIE kernel calculates the argmin of a given bfloat16 vector, returning a single int32 value for the minimum argument.", 2472),
    ("avgpool1d_bfloat16", "A kernel that performs a 1D average pooling operation on an input vector with window_size and stride as runtime parameters.", 55895),
    ("avgpool1d_relu_bfloat16", "A kernel that performs 1D average pooling with a 4-element window and stride of 4 on a 256-element bfloat16 vector, followed by elementwise ReLU activation, writing the results to a 64-element bfloat16 output buffer.", 51238),
    ("avgpool2d_bfloat16", "A kernel that performs a two dimension average pooling on an input matrix (32x32), buffer_in, with a 2x2 kernel and stride of two. The output is written to buffer_out.", 192822),
    ("avgpool2d_relu_bfloat16", "A kernel that performs 2D average pooling with ReLU activation on an input matrix (32x32), using a 2x2 window and stride 2. The output is a 16x16 matrix written to the output buffer. All operations use bfloat16.", 206612),
    ("bitcount_int16", "This AIE kernel counts the number of 1-bits (Hamming weight) in each entry of a int16_t vector, writing out a int16_t vector of the same length where each entry holds the bit count for the corresponding input value.", 42075),
    ("cast_bfloat16_to_float32", "This AIE kernel casts a bfloat16 input vector of length 512 into an output vector of float32, performing an elementwise conversion.", 2183),
    ("cast_bfloat16_to_int8", "This AIE kernel casts each element of a bfloat16 vector input to int8, performing rounding to nearest integer and saturating to the int8 value range [-128, 127]. The output is a vector of int8.", 122022),
    ("cast_float32_to_bfloat16", "This AIE kernel casts a float32 input vector of length 256 to bfloat16, writing results to the output buffer.", 3079),
    ("cast_int8_to_int32", "This AIE kernel casts each element of a signed int8 input vector to signed int32, storing the result in the output buffer. The casting should preserve the sign.", 2311),
    ("ceil_bfloat16", "This AIE kernel implements ceil operation on a bfloat16 vector, rounding each element up to the nearest integer", 52200),
    ("compare_equal_bfloat16", "This AIE kernel compares two bfloat16 vectors elementwise for equality, and writes a uint8_t mask output (1 if equal, 0 if not).", 4167),
    ("compare_equal_int32", "This AIE kernel takes two int32 input vectors and compares them element-wise for equality, producing a uint8 output vector: 1 if equal, 0 if not.", 2887),
    ("compare_gt_int8", "This AIE kernel compares two int8 input vectors of length 256 and produces an int8 output vector, where each output element is 1 if the corresponding element of the first input is greater than that of the second, and 0 otherwise.", 2758),
    ("compare_lt_int8", "This AIE kernel compares two int8_t input vectors elementwise (length 256), outputting 1 at index i if input_a[i] < input_b[i], else 0. Results are int8_t.", 2758),
    ("complexabs_bfloat16", "This AIE kernel computes the absolute value of complex numbers. real_vector contains the real parts, and imag_vector contains the imaginary parts. The result is stored in output_vector.", 79823),
    ("conv1d_bfloat16", "A kernel that performs a 1D convolution operation on a bfloat16 input vector with a bfloat16 kernel and given stride (runtime parameter). All vectors are 256-wide and the kernel is size 2.", 131645),
    ("conv1d_bias_relu_bfloat16", "A 1D convolution kernel operating on a 256-length bfloat16 vector input using a 4-element bfloat16 kernel, bfloat16 bias, and reported stride (runtime parameter). The sum is computed, biased, ReLU applied, and written to the output buffer. Output size = (256-4)//stride+1.", 284097),
    ("conv1d_int32", "A kernel that performs a 1D convolution operation on an input vector with a given kernel and stride as runtime parameters.", 6155),
    ("conv1d_k2_s1_bias_relu_bfloat16", "A kernel that performs 1D convolution with kernel size 2 and stride 1 on a bfloat16 input vector, adds a bfloat16 bias, applies ReLU, and writes bfloat16 output. The input is 256-wide and the kernel has size 2.", 190654),
    ("conv1d_k2_s2_bias_int16", "A kernel that performs a 1D convolution with kernel size 2 and stride 2 on an int16 vector with an int16 kernel and adds a scalar int16 bias to the accumulator before writing the result. Input vector is 128 elements (int16), kernel is length 2 (int16), and bias is scalar int16. The output is ((128 - 2) // 2) + 1 = 64 elements (int16), with result saturated to int16 range.", 920),
    ("conv1d_k4_s1_bias_relu_bfloat16", "A kernel that performs a 1D convolution with a bfloat16 vector of length 256. Uses a convolution kernel of size 4, stride 1, adds a bfloat16 bias after convolution, applies ReLU, and writes bfloat16 output. Output vector is length 253. ", 77098),
    ("cumsum_int16", "This AIE kernel computes the cumulative sum (cumsum) of a 256-element int16_t input vector. The result is written to the output buffer, with saturation to int16_t range on overflow.", 4423),
    ("dotproduct_bias_relu_int8", "This kernel computes the dot product of two int8 vectors of length 256, adds a scalar int32 bias, applies ReLU, clamps the result to int8_t range, and writes the result to a single-element int8_t output buffer.", 1424),
    ("dotproduct_int32", "An AIE kernel that performs a dot product operation on two int32 input vectors of the same size.", 3207),
    ("elementwise_max_int8", "This AIE kernel computes the elementwise maximum between two int8 vectors of length 512 and places the result in the output buffer.", 3270),
    ("elementwise_min_bfloat16", "This AIE kernel performs an elementwise minimum operation between two input bfloat16 vectors, writing the minimum of each pair to the output buffer.", 1991),
    ("elementwise_min_int8", "This AIE kernel performs an elementwise minimum operation between two input int8 vectors of size 256, writing the minimum of each pair to the output buffer.", 3270),
    ("gather_bfloat16_int32idx", "This AIE kernel implements a gather operation: for each idx in the int32 indices buffer, utput_data[i] = input_data[indices[i]], all with bfloat16 inputs/outputs and int32 indices. Vectors are length 256.", 3591),
    ("hardsigmoid_int8", "This AIE kernel applies the hardsigmoid activation function on a vector of int8 values. The hardsigmoid is computed as: y = max(0, min(1, x/6 + 0.5)), and the result is rescaled and quantized into int8 output with the range [0,127].", 229169),
    ("matmul_16x16_int8", "This kernel performs matrix multiplication between two 16x16 int8_t matrices A and B (row-major, shape=(16,16)), writing the result (clamped to int8_t) into a 16x16 int8_t output buffer (row-major).", 25009),
    ("matrixvector_mult_int8", "A kernel that computes the matrix-vector multiplication of an int8 matrix (16x16, row-major) and an int8 vector (length 16). Result is an int32 vector of length 16.", 977),
    ("maxpool2d_relu_int8", "This AIE kernel performs 2D max pooling with a 2x2 window and stride=2 on a 16x16 input int8 tensor. After max-pooling, ReLU is applied (all negative values are set to zero). The result is written to a 8x8 output tensor.", 1807),
    ("negate_int8", "This AIE kernel negates each element in a vector of int8_t (length 256). The input is a buffer of 256 int8, and the output buffer receives the elementwise negation.", 2503),
    ("pad1d_int32", "A kernel that performs a 1D padding operation on an input vector with given padding size and padding value as runtime parameters.", 4807),
    ("pad2d_int32", "A kernel that performs a 2D padding operation on an input matrix with given padding size and padding value as runtime parameters.", 5560),
    ("reduce_add_relu_int8", "This AIE kernel sums all 128 elements of an int8_t input buffer, applies a ReLU (sets negative sums to zero), and outputs the saturated result as a single int8_t to out_buffer.", 392),
    ("reducemin_bfloat16", "This AIE kernel calculates the min of an input bfloat16 vector and outputs the result to an output buffer.", 448),
    ("relu_int8", "This AIE kernel performs a ReLU activation on an int8 input vector of size num_elements.", 11015),
    ("shuffle_int32", "This AIE kernel performs an in-place Fisher-Yates shuffle of a length-256 int32 vector, using a runtime-provided seed parameter. The input is copied to output and then shuffled.", 15388),
    ("sigmoid_bfloat16", "This AIE kernel implements the sigmoid activation function on a bfloat16 vector", 110049),
    ("sign_int8", "This AIE kernel computes the elementwise sign of a vector of int8 values. For each input value: if positive return 1, if negative return -1, otherwise 0.", 7640),
    ("sin_bfloat16", "This AIE kernel implements the sine function on a bfloat16 vector", 70801),
    ("sqrt_bfloat16", "A kernel that computes the elementwise square root of a bfloat16 vector. Negative values are clamped to 0.0 in the output.", 150329),
    ("tanh_bfloat16", "This AIE kernel implements the tanh activation function on a bfloat16 vector. ", 579808),
    ("vectoradd_bfloat16", "This AIE kernel computes the elementwise addition of two bfloat16 input vectors of size 256, writing the result to an output bfloat16 vector.", 3079),
    ("vectoradd_int16", "This is the implementation of an AIE kernel that performs a vector addition of 2 input vectors and stores the result in 1 output vector.", 2631),
    ("vectoradd_relu_bfloat16", "This AIE kernel computes the elementwise vector addition of two bfloat16 vectors of length 256, followed by the ReLU activation. Each output is given by out[i] = max(0, in0[i] + in1[i]).", 6983),
    ("vectormatrix_mult_int32", "A kernel that performs vector-matrix multiplication of a vector with shape (16,) and matrix of (16,16).", 3141),
    ("vectorsubtract_bfloat16", "This AIE kernel performs elementwise subtraction between two bfloat16 input vectors (of length 256), writing the per-element difference to the output buffer.", 3079),
    ("vectorsubtract_int8", "This is the implementation of an AIE kernel that performs a vector subtraction of 2 input vectors and stores the result in 1 output vector.", 5510),
]

def get_npueval_kernel_path(kernel_name):
    """Get the kernel function path for a given NPUEval kernel."""
    return f"npueval_dataset/{kernel_name}/kernel_func.cc"

def get_npueval_testbench_path(kernel_name):
    """Get the testbench path for a given NPUEval kernel."""
    return f"npueval_dataset/{kernel_name}/test.py"

def get_npueval_kernels():
    """Get all NPUEval kernel names."""
    return [kernel[0] for kernel in NPUEVAL_KERNELS]

def get_npueval_kernel_descriptions():
    """Get kernel descriptions as a dictionary."""
    return {kernel[0]: kernel[1] for kernel in NPUEVAL_KERNELS}

def get_npueval_kernel_paths():
    """Get kernel paths as a dictionary."""
    return {kernel[0]: get_npueval_kernel_path(kernel[0]) for kernel in NPUEVAL_KERNELS}

def get_npueval_testbench_paths():
    """Get testbench paths as a dictionary."""
    return {kernel[0]: get_npueval_testbench_path(kernel[0]) for kernel in NPUEVAL_KERNELS}

def get_npueval_scalar_times():
    """Get scalar times as a dictionary."""
    return {kernel[0]: kernel[2] for kernel in NPUEVAL_KERNELS}
