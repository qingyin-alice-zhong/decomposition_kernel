// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void vectormatrix_mult_int32(std::int32_t vector[16], std::int32_t matrix[256], std::int32_t result[16]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"