// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

#include <cstdint>

void conv1d_k2_s2_bias_int16(std::int16_t in_buffer[128], std::int16_t out_buffer[64], std::int32_t param[3]) {
    event0();
    // Unpack parameters: kernel[0], kernel[1], bias
    std::int16_t kernel[2] = {(std::int16_t)param[0], (std::int16_t)param[1]};
    std::int16_t bias = (std::int16_t)param[2];
    
    constexpr std::int32_t VECTOR_SIZE = 128;
    constexpr std::int32_t KERNEL_SIZE = 2;
    constexpr std::int32_t STRIDE = 2;
    std::uint32_t num_windows = (VECTOR_SIZE - KERNEL_SIZE) / STRIDE + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        std::int16_t acc = bias;
        for (uint32_t j = 0; j < KERNEL_SIZE; j++) {
            acc += in_buffer[i * STRIDE + j] * kernel[j];
        }
        if (acc >  32767) acc =  32767;
        if (acc < -32768) acc = -32768;
        out_buffer[i] = acc;
    }
    event1();
}

} // extern "C"