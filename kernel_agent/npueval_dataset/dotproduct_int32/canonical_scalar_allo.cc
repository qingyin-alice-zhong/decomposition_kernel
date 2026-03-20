// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void dotproduct_int32(std::int32_t vector1[256], std::int32_t vector2[256], std::int32_t result[1]) {
    event0();
    constexpr std::int32_t vector_size = 256;
    result[0] = 0;
    for (std::uint32_t i = 0; i < vector_size; i++) {
        result[0] += vector1[i] * vector2[i];
    }
    event1();
}

} // extern "C"