// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void cast_bfloat16_to_float32(bfloat16 in_buffer[512], float out_buffer[512]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"