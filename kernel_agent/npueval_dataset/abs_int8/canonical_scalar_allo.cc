// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void abs_int8(std::int8_t in_buffer[1024], std::int8_t out_buffer[1024]) {
    event0();
    constexpr std::int32_t nbytes = 1024;
    for (std::uint32_t i = 0; i < nbytes; i++) {
        std::int8_t value = in_buffer[i];
        if (value < 0) {
            out_buffer[i] = -value;
        } else {
            out_buffer[i] = value;
        }
    }
    event1();
}

} // extern "C"