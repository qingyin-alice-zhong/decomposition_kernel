// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv1d_int32(std::int32_t in_buffer[256], std::int32_t out_buffer[254], std::int32_t param[4]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    constexpr std::int32_t kernel_size = 3;
    // Extract kernel and stride from parameter array
    std::int32_t kernel[3] = {param[0], param[1], param[2]};
    std::int32_t stride_val = param[3];
    std::int32_t num_windows = (vector_size - kernel_size) / stride_val + 1;

    for (std::int32_t i = 0; i < num_windows; i++) {
        std::int32_t conv_sum = 0;
        for (std::int32_t j = 0; j < kernel_size; j++) {
            conv_sum += in_buffer[i * stride_val + j] * kernel[j];
        }
        out_buffer[i] = conv_sum;
    }
    event1();
}

} // extern "C"