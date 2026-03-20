// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void compare_equal_bfloat16(bfloat16 input_a[256], bfloat16 input_b[256], bfloat16 output_mask[256]) {
    event0();
    constexpr std::uint32_t vector_size = 256;
    for (std::uint32_t i = 0; i < vector_size; i++) {
        output_mask[i] = (input_a[i] == input_b[i]) ? bfloat16(1.0f) : bfloat16(0.0f);
    }
    event1();
}

} // extern "C"