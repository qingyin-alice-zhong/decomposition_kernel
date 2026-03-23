// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_3x64_b1a_fp32(float input[300], float output[512], float param[224]) {
    constexpr int IN_CHANNELS = 3;
    constexpr int OUT_CHANNELS = 8;
    constexpr int INPUT_HEIGHT = 10;
    constexpr int INPUT_WIDTH = 10;
    constexpr int OUTPUT_HEIGHT = 8;
    constexpr int OUTPUT_WIDTH = 8;
    constexpr int KERNEL_H = 3;
    constexpr int KERNEL_W = 3;
    constexpr int WEIGHT_SIZE = OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W;

    event0();

    const float *__restrict in = input;
    const float *__restrict weights = param;
    const float *__restrict bias = param + WEIGHT_SIZE;
    float *__restrict out = output;

    for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
        for (int oh = 0; oh < OUTPUT_HEIGHT; ++oh) {
            for (int ow = 0; ow < OUTPUT_WIDTH; ++ow) {
                float acc = bias[oc];
                for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                    for (int kh = 0; kh < KERNEL_H; ++kh) {
                        const int ih = oh + kh;
                        const float *__restrict input_row = in + (ic * INPUT_HEIGHT + ih) * INPUT_WIDTH + ow;
                        const float *__restrict weight_row = weights + ((oc * IN_CHANNELS + ic) * KERNEL_H + kh) * KERNEL_W;
                        for (int kw = 0; kw < KERNEL_W; ++kw) {
                            acc += input_row[kw] * weight_row[kw];
                        }
                    }
                }
                const int output_idx = (oc * OUTPUT_HEIGHT + oh) * OUTPUT_WIDTH + ow;
                out[output_idx] = acc;
            }
        }
    }

    event1();
}

} // extern "C"
