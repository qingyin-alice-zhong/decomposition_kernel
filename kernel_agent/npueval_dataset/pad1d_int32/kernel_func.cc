// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void pad1d_int32(std::int32_t input_vector[256], std::int32_t padded_vector[260], std::int32_t param[2]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"