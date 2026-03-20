// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void hardsigmoid_int8(const std::int8_t input_vector[256], std::int8_t output_vector[256]) {
    event0();
    constexpr std::int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float x = (float)input_vector[i];
        float y = (x / 6.0f) + 0.5f;
        if (y < 0.0f) y = 0.0f;
        if (y > 1.0f) y = 1.0f;
        std::int32_t out = (std::int32_t)(y * 127.0f + 0.5f);
        if (out > 127) out = 127;
        if (out < 0) out = 0;
        output_vector[i] = (std::int8_t)out;
    }
    event1();
}

} // extern "C"