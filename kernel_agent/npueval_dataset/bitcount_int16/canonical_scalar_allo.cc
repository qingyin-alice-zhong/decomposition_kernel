// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void bitcount_uint16(std::int16_t input_vector[256], std::int16_t output_vector[256]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        std::uint16_t val = input_vector[i];
        std::uint16_t count = 0;
        while (val) {
            count += val & 1;
            val >>= 1;
        }
        output_vector[i] = count;
    }
    event1();
}

} // extern "C"