// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void reducemin_bfloat16(bfloat16 in_buffer[64], float32 out_buffer[1]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"