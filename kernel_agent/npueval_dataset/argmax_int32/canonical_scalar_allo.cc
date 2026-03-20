// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void argmax_int32(std::int32_t input_vector[256], std::int32_t result[1]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    std::int32_t max_index = 0;
    std::int32_t max_value = input_vector[0];

    for (std::int32_t i = 1; i < vector_size; i++) {
        if (input_vector[i] > max_value) {
            max_value = input_vector[i];
            max_index = i;
        }
    }

    result[0] = max_index;
    event1();
}

} // extern "C"