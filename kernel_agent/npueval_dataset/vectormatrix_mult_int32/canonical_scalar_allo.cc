// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void vectormatrix_mult_int32(std::int32_t vector[16], std::int32_t matrix[256], std::int32_t result[16]) {
    event0();
    constexpr std::uint32_t vector_size = 16;
    constexpr std::uint32_t matrix_cols = 16;
    for (uint32_t i = 0; i < matrix_cols; i++) {
        result[i] = 0;
    }
    for (uint32_t i = 0; i < matrix_cols; i++) {
        for (uint32_t j = 0; j < vector_size; j++) {
            result[i] += vector[j] * matrix[j * matrix_cols + i];
        }
    }
    event1();
}

} // extern "C"