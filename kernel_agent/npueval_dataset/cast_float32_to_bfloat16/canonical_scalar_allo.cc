// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void cast_float32_to_bfloat16(float in_buffer[256], bfloat16 out_buffer[256]) {
    event0();
    constexpr std::int32_t N = 256;
    for (int i = 0; i < N; ++i) {
        out_buffer[i] = bfloat16(in_buffer[i]);
    }
    event1();
}

} // extern "C"