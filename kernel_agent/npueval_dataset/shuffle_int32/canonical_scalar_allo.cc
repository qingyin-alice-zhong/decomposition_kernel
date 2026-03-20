// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void shuffle_int32(std::int32_t input[256], std::int32_t output[256], std::int32_t seed[1]) {
    event0();
    constexpr std::int32_t N = 256;

    for (int i = 0; i < N; ++i) {
        output[i] = input[i];
    }

    uint32_t state = static_cast<uint32_t>(seed[0]);

    for (int i = N - 1; i > 0; --i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        int j = static_cast<int>(state % uint32_t(i + 1));
        int32_t tmp = output[i];
        output[i] = output[j];
        output[j] = tmp;
    }
    event1();
}

} // extern "C"