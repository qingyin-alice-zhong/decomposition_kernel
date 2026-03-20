// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void pad1d_int32(std::int32_t input_vector[256], std::int32_t padded_vector[260], std::int32_t param[2]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    std::int32_t pad_size_val = param[0];
    std::int32_t pad_value_val = param[1];
    std::int32_t padded_length = vector_size + 2 * pad_size_val;

    for (std::int32_t i = 0; i < padded_length; i++) {
        padded_vector[i] = pad_value_val;
    }

    for (std::int32_t i = 0; i < vector_size; i++) {
        padded_vector[i + pad_size_val] = input_vector[i];
    }
    event1();
}

} // extern "C"