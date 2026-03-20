// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void complexabs_bfloat16(bfloat16 real_vector[64], bfloat16 imag_vector[64], bfloat16 output_vector[64]) {
    event0();
    constexpr std::int32_t vector_size = 64;
    for (uint32_t i = 0; i < vector_size; i++) {
        float real = (float)real_vector[i];
        float imag = (float)imag_vector[i];
        float x = real * real + imag * imag;
        if (x == 0.0f) {
            output_vector[i] = (bfloat16)0.0f;
            continue;
        }
        float xhalf = 0.5f * x;
        std::int32_t j = *(std::int32_t*)&x;
        j = 0x5f3759df - (j >> 1); // fast inverse square root
        float y = *(float*)&j;
        y = y * (1.5f - xhalf * y * y);
        float result = x * y;
        output_vector[i] = (bfloat16)result;
    }
    event1();
}

} // extern "C"