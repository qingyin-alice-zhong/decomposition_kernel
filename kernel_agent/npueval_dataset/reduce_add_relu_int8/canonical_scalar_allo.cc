// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void reduce_add_relu_int8(std::int8_t in_buffer[128], std::int32_t out_buffer[1]) {
    event0();
    constexpr std::int32_t NUM_ELEMENTS = 128;
    std::int32_t sum = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        sum += in_buffer[i];
    }
    if (sum < 0)
        sum = 0;
    if (sum > 127) sum = 127;
    if (sum < -128) sum = -128;
    out_buffer[0] = sum;
    event1();
}

} // extern "C"