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
    // TODO: Implement the kernel
    event1();
}

} // extern "C"