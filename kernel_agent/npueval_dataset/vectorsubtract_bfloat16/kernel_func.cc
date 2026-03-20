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
    // TODO: Implement the kernel
    event1();
}

} // extern "C"