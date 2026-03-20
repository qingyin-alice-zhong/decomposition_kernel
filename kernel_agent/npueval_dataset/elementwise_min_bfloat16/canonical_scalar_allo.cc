// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void elementwise_min_bfloat16(bfloat16 in_buffer_a[256], bfloat16 in_buffer_b[256], bfloat16 out_buffer[256]) {
    event0();
    constexpr std::int32_t VEC_SIZE = 256;
    for (int i = 0; i < VEC_SIZE; ++i) {
        float a = (float)in_buffer_a[i];
        float b = (float)in_buffer_b[i];
        out_buffer[i] = (a < b) ? in_buffer_a[i] : in_buffer_b[i];
    }
    event1();
}

} // extern "C"