// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void pad2d_int32(std::int32_t input_matrix[256], std::int32_t padded_matrix[400], std::int32_t param[2]) {
    event0();
    // Unpack parameters
    std::uint32_t pad_size = static_cast<std::uint32_t>(param[0]);
    std::int32_t pad_value = param[1];
    
    constexpr std::uint32_t input_rows = 16;
    constexpr std::uint32_t input_cols = 16;
    std::uint32_t padded_rows = input_rows + 2 * pad_size;
    std::uint32_t padded_cols = input_cols + 2 * pad_size;

    // Initialize with pad_value
    for (uint32_t i = 0; i < padded_rows; i++) {
        for (uint32_t j = 0; j < padded_cols; j++) {
            padded_matrix[i * padded_cols + j] = pad_value;
        }
    }

    // Copy input data to center
    for (uint32_t i = 0; i < input_rows; i++) {
        for (uint32_t j = 0; j < input_cols; j++) {
            padded_matrix[(i + pad_size) * padded_cols + (j + pad_size)] = input_matrix[i * input_cols + j];
        }
    }
    event1();
}

} // extern "C"