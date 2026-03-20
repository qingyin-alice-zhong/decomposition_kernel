// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void vectorsubtract_bfloat16(bfloat16 input_a[256], bfloat16 input_b[256], bfloat16 output[256]) {
    event0();
    constexpr std::int32_t NUM_ELEMENTS = 256;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        output[i] = input_a[i] - input_b[i];
    }
    event1();
}

} // extern "C"