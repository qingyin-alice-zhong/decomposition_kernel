// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void elementwise_max_int8(std::int8_t in_buffer1[256], std::int8_t in_buffer2[256], std::int8_t out_buffer[256]) {
    event0();
    constexpr std::int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        std::int8_t a = in_buffer1[i];
        std::int8_t b = in_buffer2[i];
        out_buffer[i] = (a > b) ? a : b;
    }
    event1();
}

} // extern "C"