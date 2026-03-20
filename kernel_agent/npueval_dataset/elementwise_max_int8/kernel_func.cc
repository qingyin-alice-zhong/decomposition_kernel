// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void elementwise_max_int8(std::int8_t in_buffer1[256], std::int8_t in_buffer2[256], std::int8_t out_buffer[256]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"