// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void vectoradd_int16(std::int16_t in_buffer1[256], std::int16_t in_buffer2[256], std::int16_t out_buffer[256]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    for (std::int32_t i = 0; i < vector_size; ++i) {
        out_buffer[i] = in_buffer1[i] + in_buffer2[i];
    }
    event1();
}

} // extern "C"
