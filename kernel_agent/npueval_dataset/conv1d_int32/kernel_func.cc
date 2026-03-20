// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv1d_int32(std::int32_t in_buffer[256], std::int32_t out_buffer[254], std::int32_t param[4]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"