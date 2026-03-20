// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_3x64_b1a_fp32(float input[972], float output[2048], float param[224]) {
    constexpr int IN_CHANNELS = 3;
    constexpr int OUT_CHANNELS = 8;
    constexpr int INPUT_HEIGHT = 18;
    constexpr int INPUT_WIDTH = 18;
    constexpr int OUTPUT_HEIGHT = 16;
    constexpr int OUTPUT_WIDTH = 16;
    constexpr int KERNEL_H = 3;
    constexpr int KERNEL_W = 3;
    constexpr int WEIGHT_SIZE = OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W;
    constexpr int BIAS_SIZE = OUT_CHANNELS;
    constexpr int INPUT_SIZE = IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    constexpr int OUTPUT_SIZE = OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH;

    event0();

    float input_local[INPUT_SIZE];
    float weight_local[WEIGHT_SIZE];
    float bias_local[BIAS_SIZE];

    for (int i = 0; i < INPUT_SIZE; ++i) {
        input_local[i] = input[i];
    }
    for (int i = 0; i < WEIGHT_SIZE; ++i) {
        weight_local[i] = param[i];
    }
    for (int i = 0; i < BIAS_SIZE; ++i) {
        bias_local[i] = param[WEIGHT_SIZE + i];
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = 0.0f;
    }

    for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
        for (int oh = 0; oh < OUTPUT_HEIGHT; ++oh) {
            for (int ow = 0; ow < OUTPUT_WIDTH; ++ow) {
                float acc = bias_local[oc];
                for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                    for (int kh = 0; kh < KERNEL_H; ++kh) {
                        const int ih = oh + kh;
                        const int input_row_offset = (ic * INPUT_HEIGHT + ih) * INPUT_WIDTH;
                        const int weight_row_offset = ((oc * IN_CHANNELS + ic) * KERNEL_H + kh) * KERNEL_W;
                        for (int kw = 0; kw < KERNEL_W; ++kw) {
                            const int iw = ow + kw;
                            acc += input_local[input_row_offset + iw] * weight_local[weight_row_offset + kw];
                        }
                    }
                }
                const int output_idx = (oc * OUTPUT_HEIGHT + oh) * OUTPUT_WIDTH + ow;
                output[output_idx] = acc;
            }
        }
    }

    event1();
}

} // extern "C"
