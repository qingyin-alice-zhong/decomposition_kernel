// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void reducemin_bfloat16(bfloat16 in_buffer[64], float out_buffer[1]) {
    event0();
    constexpr std::int32_t num_elements = 64;
    bfloat16 min_value = in_buffer[0];
    for (int i = 1; i < num_elements; i++) {
        if (min_value > in_buffer[i]) {
            min_value = in_buffer[i];
        }
    }
    out_buffer[0] = min_value;
    event1();
}

} // extern "C"