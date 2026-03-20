// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void gather_bfloat16_int32idx(bfloat16 input_data[256], std::int32_t indices[256], bfloat16 output_data[256]) {
    event0();
    constexpr std::int32_t GATHER_N = 256;
    for (int i = 0; i < GATHER_N; i++) {
        std::int32_t idx = indices[i];
        output_data[i] = input_data[idx];
    }
    event1();
}

} // extern "C"