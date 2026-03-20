// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

#include <aie_api/aie.hpp>
#include <cstdint>

void cast_bfloat16_to_int8(bfloat16 input[256], std::int8_t output[256]) {
    event0();
    constexpr int N = 256;
    for (int i = 0; i < N; ++i) {
        float x = float(input[i]);
        int xi = int(x);
        int floor_x;
        if (x >= 0.0f) {
            floor_x = xi;
        } else {
            if (float(xi) == x) {
                floor_x = xi;
            } else {
                floor_x = xi - 1;
            }
        }
        float frac = x - float(floor_x);
        int r;
        if (frac < 0.5f) {
            r = floor_x;
        } else if (frac > 0.5f) {
            r = floor_x + 1;
        } else {
            r = (floor_x & 1) == 0 ? floor_x : (floor_x + 1);
        }
        if (r > 127)      r = 127;
        else if (r < -128) r = -128;
        output[i] = std::int8_t(r);
    }
    event1();
}

} // extern "C"