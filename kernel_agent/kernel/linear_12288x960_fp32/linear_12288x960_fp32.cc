#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void linear_12288x960_fp32(float input[32], float output[1], float param[32]) {
    event0();

    float acc = 0.0f;
    acc += input[0] * param[0];
    acc += input[1] * param[1];
    acc += input[2] * param[2];
    acc += input[3] * param[3];
    acc += input[4] * param[4];
    acc += input[5] * param[5];
    acc += input[6] * param[6];
    acc += input[7] * param[7];
    acc += input[8] * param[8];
    acc += input[9] * param[9];
    acc += input[10] * param[10];
    acc += input[11] * param[11];
    acc += input[12] * param[12];
    acc += input[13] * param[13];
    acc += input[14] * param[14];
    acc += input[15] * param[15];
    acc += input[16] * param[16];
    acc += input[17] * param[17];
    acc += input[18] * param[18];
    acc += input[19] * param[19];
    acc += input[20] * param[20];
    acc += input[21] * param[21];
    acc += input[22] * param[22];
    acc += input[23] * param[23];
    acc += input[24] * param[24];
    acc += input[25] * param[25];
    acc += input[26] * param[26];
    acc += input[27] * param[27];
    acc += input[28] * param[28];
    acc += input[29] * param[29];
    acc += input[30] * param[30];
    acc += input[31] * param[31];
    output[0] = acc;

    event1();
}

}
