// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void elementwise_min_int8(std::int8_t input1[256], std::int8_t input2[256], std::int8_t output[256]) {
    event0();
    constexpr std::int32_t N = 256;
    for (int i = 0; i < N; ++i) {
        output[i] = (input1[i] < input2[i]) ? input1[i] : input2[i];
    }
    event1();
}

} // extern "C"