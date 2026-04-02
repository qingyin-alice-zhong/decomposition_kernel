#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void linear_32x720_fp32(float input[32], float output[16], float param[528]) {
    constexpr int OUT_TILE = 16;

    event0();

    (void)input;

    for (int o = 0; o < OUT_TILE; ++o) {
        output[o] = param[o];
    }

    event1();
}

}
