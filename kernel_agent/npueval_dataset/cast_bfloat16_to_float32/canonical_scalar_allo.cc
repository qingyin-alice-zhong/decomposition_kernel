// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void cast_bfloat16_to_float32(bfloat16 in_buffer[512], float out_buffer[512]) {
    event0();
    constexpr std::int32_t N = 512;
    for (int i = 0; i < N; i++) {
        out_buffer[i] = float(in_buffer[i]);
    }
    event1();
}

} // extern "C"