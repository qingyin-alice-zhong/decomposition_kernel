// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void negate_int8(std::int8_t input_vector[256], std::int8_t output_vector[256]) {
    event0();
    constexpr std::uint32_t VECTOR_SIZE = 256;
    for (std::uint32_t i = 0; i < VECTOR_SIZE; ++i) {
        output_vector[i] = -input_vector[i];
    }
    event1();
}

} // extern "C"
