// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void dotproduct_bias_relu_int8(std::int8_t in0[256], std::int32_t out[1], std::int32_t param[257]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"