#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void linear_32x720_fp32(float input[32], float output[8], float param[264]) {
    constexpr int OUT_TILE = 4;
    constexpr int ABI_TILE = 8;

    event0();

    for (int o = 0; o < OUT_TILE; ++o) {
        output[o] = param[o];
    }

    for (int o = OUT_TILE; o < ABI_TILE; ++o) {
        output[o] = 0.0f;
    }

    event1();
}

}
