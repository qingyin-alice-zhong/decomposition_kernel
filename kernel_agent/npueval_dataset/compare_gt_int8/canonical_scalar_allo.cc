// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void compare_gt_int8(std::int8_t in_buffer0[256], std::int8_t in_buffer1[256], std::int8_t out_buffer[256]) {
    event0();
    constexpr std::int32_t N = 256;
    for (int i = 0; i < N; i++) {
        out_buffer[i] = in_buffer0[i] > in_buffer1[i] ? 1 : 0;
    }
    event1();
}

} // extern "C"