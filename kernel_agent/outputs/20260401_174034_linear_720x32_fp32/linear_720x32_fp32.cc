#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void linear_720x32_fp32(float input[32], float output[1], float param[33]) {
    constexpr int IN_FEATURES = 32;

    event0();

    const float* weight = param;
    const float* bias = param + IN_FEATURES;

    float acc = bias[0];
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    acc += input[4] * weight[4];
    acc += input[5] * weight[5];
    acc += input[6] * weight[6];
    acc += input[7] * weight[7];
    acc += input[8] * weight[8];
    acc += input[9] * weight[9];
    acc += input[10] * weight[10];
    acc += input[11] * weight[11];
    acc += input[12] * weight[12];
    acc += input[13] * weight[13];
    acc += input[14] * weight[14];
    acc += input[15] * weight[15];
    acc += input[16] * weight[16];
    acc += input[17] * weight[17];
    acc += input[18] * weight[18];
    acc += input[19] * weight[19];
    acc += input[20] * weight[20];
    acc += input[21] * weight[21];
    acc += input[22] * weight[22];
    acc += input[23] * weight[23];
    acc += input[24] * weight[24];
    acc += input[25] * weight[25];
    acc += input[26] * weight[26];
    acc += input[27] * weight[27];
    acc += input[28] * weight[28];
    acc += input[29] * weight[29];
    acc += input[30] * weight[30];
    acc += input[31] * weight[31];
    output[0] = acc;

    event1();
}

}
