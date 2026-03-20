// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void cast_int8_to_int32(std::int8_t input_vector[256], std::int32_t output_vector[256]) {
    event0();
    constexpr std::int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        output_vector[i] = (std::int32_t)input_vector[i];
    }
    event1();
}

} // extern "C"