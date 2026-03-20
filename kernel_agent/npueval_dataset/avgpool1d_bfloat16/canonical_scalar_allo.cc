// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void avgpool1d_bfloat16(bfloat16 in_buffer[1024], bfloat16 out_buffer[512], 
                       std::int32_t param[2]) {
    event0();
    constexpr std::uint32_t num_elements = 1024;
    std::uint32_t ws = param[0];
    std::uint32_t st = param[1];
    std::uint32_t num_windows = (num_elements - ws) / st + 1;

    for (std::uint32_t i = 0; i < num_windows; i++) {
        float sum = 0.0f;
        for (std::uint32_t j = 0; j < ws; j++) {
            float current_val = (float)*(in_buffer + i * st + j);
            sum += current_val;
        }
        float avg = sum / ws;
        out_buffer[i] = (bfloat16)avg;
    }
    event1();
}

} // extern "C"