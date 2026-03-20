// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void sign_int8(std::int8_t input_vector[256], std::int8_t output_vector[256]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
        std::int8_t val = input_vector[i];
        if (val > 0)
            output_vector[i] = 1;
        else if (val < 0)
            output_vector[i] = -1;
        else
            output_vector[i] = 0;
    }
    event1();
}

} // extern "C"