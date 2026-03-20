// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void add_offset_int8(std::int8_t in_buffer[256], std::int8_t out_buffer[256], std::int32_t offset[1]) {
    event0();

    // Define the vectorization factor. For int8_t, 32 elements correspond to 256 bits,
    // which is an optimal access size on AIE-ML/XDNA 1 (as per documentation for int8, 32 is a native size).
    constexpr int VEC_SIZE = 32; 
    constexpr int NUM_VECTORS = 256 / VEC_SIZE; // 256 / 32 = 8 iterations

    // Extract the scalar offset value from the input array.
    std::int32_t scalar_offset = offset[0];

    // Use restrict pointers for better compiler optimization.
    std::int8_t *__restrict p_in = in_buffer;
    std::int8_t *__restrict p_out = out_buffer;

    // Broadcast the scalar offset to an int32_t vector.
    // This creates a vector where all 32 elements are equal to `scalar_offset`.
    aie::vector<std::int32_t, VEC_SIZE> v_offset_int32 = aie::broadcast<std::int32_t, VEC_SIZE>(scalar_offset);
    
    // Convert the int32_t vector to an acc32 accumulator.
    // This accumulator will hold the offset value, ready for addition with input values.
    // `acc32` provides 32-bit precision, which is sufficient for intermediate results
    // when adding int8_t values to an int32_t offset.
    aie::accum<acc32, VEC_SIZE> acc_offset_vec;
    // The `from_vector` method with a shift of 0 performs an integer conversion,
    // effectively promoting the 32-bit values to the accumulator's internal representation.
    acc_offset_vec.from_vector(v_offset_int32, 0);

    // Explicitly unroll the loop. For a small, fixed number of iterations (8 in this case),
    // full unrolling can significantly improve performance by:
    // 1. Eliminating loop control overhead (branching, counter updates).
    // 2. Exposing more instruction-level parallelism to the AIE compiler's scheduler,
    //    allowing it to better utilize the vector units and hide latencies.
    // This can lead to a lower cycle count compared to a pipelined loop.
    _Pragma("unroll")
    for (int i = 0; i < NUM_VECTORS; ++i) {
        // Load a vector of int8_t elements from the input buffer.
        // `aie::load_v` performs an aligned vector load. For VEC_SIZE=32 and int8_t,
        // this is a 256-bit (32-byte) load, which is efficient on AIE-ML/XDNA 1.
        aie::vector<std::int8_t, VEC_SIZE> v_in8 = aie::load_v<VEC_SIZE>(p_in);

        // Convert the loaded int8_t vector to an acc32 accumulator.
        // This step is crucial to mimic the `x.astype(np.int16)` behavior from the
        // Python reference, promoting the 8-bit input values to 32-bit accumulators.
        // This prevents overflow during the addition operation before the final saturation.
        aie::accum<acc32, VEC_SIZE> acc_val;
        acc_val.from_vector(v_in8, 0); // Shift 0 for integer conversion.

        // Perform the element-wise addition. This operation happens at the
        // higher precision of the acc32 accumulators.
        aie::accum<acc32, VEC_SIZE> acc_res = aie::add(acc_val, acc_offset_vec);

        // Convert the result accumulator back to an int8_t vector.
        // The `to_vector<std::int8_t>(0)` method automatically handles
        // rounding and saturation to the target int8_t range ([-128, 127]).
        // This directly implements the `np.clip(res, -128, 127)` behavior
        // from the reference Python code.
        aie::vector<std::int8_t, VEC_SIZE> v_out8 = acc_res.to_vector<std::int8_t>(0);

        // Store the resulting int8_t vector to the output buffer.
        // `aie::store_v` performs an aligned vector store (256-bit).
        aie::store_v(p_out, v_out8);

        // Advance the pointers for the next iteration.
        p_in += VEC_SIZE;
        p_out += VEC_SIZE;
    }

    event1();
}

} // extern "C"