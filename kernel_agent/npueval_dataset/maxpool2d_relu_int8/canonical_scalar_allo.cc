// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void maxpool2d_relu_int8(std::int8_t input[256], std::int8_t output[64]) {
    constexpr std::int32_t ROWS = 16;
    constexpr std::int32_t COLS = 16;
    constexpr std::int32_t WINDOW_SIZE = 2;
    constexpr std::int32_t STRIDE = 2;

    event0();
    for (int i = 0; i < ROWS; i += STRIDE) {
        for (int j = 0; j < COLS; j += STRIDE) {
            std::int8_t max_val = input[i * COLS + j];
            for (int wi = 0; wi < WINDOW_SIZE; wi++) {
                for (int wj = 0; wj < WINDOW_SIZE; wj++) {
                    int row_idx = i + wi;
                    int col_idx = j + wj;
                    if (row_idx < ROWS && col_idx < COLS) {
                        std::int8_t val = input[row_idx * COLS + col_idx];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
            if (max_val < 0) max_val = 0;  // ReLU
            output[(i / STRIDE) * (COLS / STRIDE) + (j / STRIDE)] = max_val;
        }
    }
    event1();
}

} // extern "C"