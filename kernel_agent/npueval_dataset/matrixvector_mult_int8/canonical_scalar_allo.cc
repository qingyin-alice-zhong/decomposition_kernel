// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void matrixvector_mult_int8(std::int8_t matrix[256], std::int8_t vector[16], std::int32_t result[16]) {
    event0();
    constexpr int ROWS = 16;
    constexpr int COLS = 16;
    for (int i = 0; i < ROWS; ++i) {
        std::int32_t acc = 0;
        for (int j = 0; j < COLS; ++j) {
            acc += static_cast<std::int32_t>(matrix[i * COLS + j]) * static_cast<std::int32_t>(vector[j]);
        }
        result[i] = acc;
    }
    event1();
}

} // extern "C"