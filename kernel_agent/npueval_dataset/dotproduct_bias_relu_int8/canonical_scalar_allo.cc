// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void dotproduct_bias_relu_int8(std::int8_t in0[256], std::int32_t out[1], std::int32_t param[257]) {
    event0();
    // Unpack parameters: in1[20] + bias[1] = param[21]
    constexpr std::int32_t N = 256;
    std::int32_t acc = 0;
    for (int i = 0; i < N; i++) {
        std::int8_t in1_val = (std::int8_t)param[i];
        acc += (std::int32_t)in0[i] * (std::int32_t)in1_val;
    }
    acc += param[20];  // bias
    if (acc < 0) acc = 0;
    if (acc > 127) acc = 127;
    if (acc < -128) acc = -128;
    out[0] = acc;
    event1();
}

} // extern "C"