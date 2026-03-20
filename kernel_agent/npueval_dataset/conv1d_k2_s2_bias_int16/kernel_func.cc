// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv1d_k2_s2_bias_int16(std::int16_t in_buffer[128], std::int16_t out_buffer[64], std::int32_t param[3]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"