// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void compare_lt_int8(std::int8_t input_a[256], std::int8_t input_b[256], std::int8_t output[256]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"