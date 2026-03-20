// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

#include <aie_api/aie.hpp>

void avgpool1d_relu_bfloat16(bfloat16 in_buffer[256], bfloat16 out_buffer[64]) {
    constexpr std::int32_t vector_size = 256;
    constexpr std::int32_t window_size = 4;
    constexpr std::int32_t stride = 4;

    event0();
    for (uint32_t i = 0; i < vector_size / stride; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < window_size; j++) {
            float val = (float)in_buffer[i * stride + j];
            sum += val;
        }
        float avg = sum / window_size;
        if (avg < 0.0f)  // ReLU
            avg = 0.0f;
        out_buffer[i] = (bfloat16)avg;
    }
    event1();
}

} // extern "C"