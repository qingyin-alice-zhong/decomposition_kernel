#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void rms_norm_960_fp32(float input[32], float output[32], float param[33]) {

    event0();

    const float inv_std = param[32];
    output[0] = input[0] * inv_std * param[0];
    output[1] = input[1] * inv_std * param[1];
    output[2] = input[2] * inv_std * param[2];
    output[3] = input[3] * inv_std * param[3];
    output[4] = input[4] * inv_std * param[4];
    output[5] = input[5] * inv_std * param[5];
    output[6] = input[6] * inv_std * param[6];
    output[7] = input[7] * inv_std * param[7];
    output[8] = input[8] * inv_std * param[8];
    output[9] = input[9] * inv_std * param[9];
    output[10] = input[10] * inv_std * param[10];
    output[11] = input[11] * inv_std * param[11];
    output[12] = input[12] * inv_std * param[12];
    output[13] = input[13] * inv_std * param[13];
    output[14] = input[14] * inv_std * param[14];
    output[15] = input[15] * inv_std * param[15];
    output[16] = input[16] * inv_std * param[16];
    output[17] = input[17] * inv_std * param[17];
    output[18] = input[18] * inv_std * param[18];
    output[19] = input[19] * inv_std * param[19];
    output[20] = input[20] * inv_std * param[20];
    output[21] = input[21] * inv_std * param[21];
    output[22] = input[22] * inv_std * param[22];
    output[23] = input[23] * inv_std * param[23];
    output[24] = input[24] * inv_std * param[24];
    output[25] = input[25] * inv_std * param[25];
    output[26] = input[26] * inv_std * param[26];
    output[27] = input[27] * inv_std * param[27];
    output[28] = input[28] * inv_std * param[28];
    output[29] = input[29] * inv_std * param[29];
    output[30] = input[30] * inv_std * param[30];
    output[31] = input[31] * inv_std * param[31];

    event1();
}

}
