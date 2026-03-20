// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void complexabs_bfloat16(bfloat16 real_vector[64], bfloat16 imag_vector[64], bfloat16 output_vector[64]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"