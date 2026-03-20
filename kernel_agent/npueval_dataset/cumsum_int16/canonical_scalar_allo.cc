// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void cumsum_int16(std::int16_t in_buffer[256], std::int16_t out_buffer[256]) {
    event0();
    constexpr int N = 256;
    std::int32_t acc = 0;
    for (int i = 0; i < N; ++i) {
        acc += static_cast<std::int32_t>(in_buffer[i]);
        // Clamp to std::int16_t range
        if (acc > 32767) acc = 32767;
        if (acc < -32768) acc = -32768;
        out_buffer[i] = static_cast<std::int16_t>(acc);
    }
    event1();
}

} // extern "C"