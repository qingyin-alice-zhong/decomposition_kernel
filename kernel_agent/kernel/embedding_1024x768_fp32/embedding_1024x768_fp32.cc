#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void embedding_1024x768_fp32(int32_t position_ids[16], float output[1024], float param[1032]) {
    constexpr int TILE_SEQ = 16;
    constexpr int TILE_HIDDEN = 64;

    event0();

    const float* table = param;

    for (int t = 0; t < TILE_SEQ; ++t) {
        (void)position_ids[t];
        for (int h = 0; h < TILE_HIDDEN; ++h) {
            const int out_idx = t * TILE_HIDDEN + h;
            output[out_idx] = table[t * TILE_HIDDEN + h];
        }
    }

    event1();
}

}