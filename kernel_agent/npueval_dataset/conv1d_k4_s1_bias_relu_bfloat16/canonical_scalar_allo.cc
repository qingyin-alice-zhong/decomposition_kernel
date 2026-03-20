// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv1d_k4_s1_bias_relu_bfloat16(bfloat16 in_buffer[256], bfloat16 out_buffer[252], std::int32_t param[5]) {
    event0();
    
    // Unpack parameters: kernel[4], bias (all stored as float32 in int32)
    bfloat16 kernel[4];
    kernel[0] = (bfloat16)(*(float*)&param[0]);
    kernel[1] = (bfloat16)(*(float*)&param[1]);
    kernel[2] = (bfloat16)(*(float*)&param[2]);
    kernel[3] = (bfloat16)(*(float*)&param[3]);
    bfloat16 bias = (bfloat16)(*(float*)&param[4]);
    
    constexpr std::int32_t VECTOR_SIZE = 256;
    constexpr std::int32_t KERNEL_SIZE = 4;
    constexpr std::int32_t STRIDE = 1;
    std::int32_t num_windows = 252;  // Fixed output size for 4-byte alignment

    for (std::int32_t i = 0; i < num_windows; i++) {
        float acc = 0.0f;
        for (std::int32_t j = 0; j < KERNEL_SIZE; j++) {
            acc += (float)in_buffer[i * STRIDE + j] * (float)kernel[j];
        }
        acc += (float)bias;
        float out_val = acc > 0.0f ? acc : 0.0f;  // ReLU
        out_buffer[i] = (bfloat16)out_val;
    }
    
    event1();
}

} // extern "C"