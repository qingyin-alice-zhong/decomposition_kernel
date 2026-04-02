#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void embedding_1024x768_fp32(int32_t position_ids[16], float output[1024], float param[1032]) {
    constexpr int TILE_SEQ = 16;
    constexpr int TILE_HIDDEN = 64;
    constexpr int TABLE_SIZE = TILE_SEQ * TILE_HIDDEN;

    event0();

    const int id_base = static_cast<int>(param[TABLE_SIZE + 0]);
    const int valid_rows = static_cast<int>(param[TABLE_SIZE + 1]);
    const float* table = param;

    for (int t = 0; t < TILE_SEQ; ++t) {
        const int row = position_ids[t] - id_base;
        for (int h = 0; h < TILE_HIDDEN; ++h) {
            const int out_idx = t * TILE_HIDDEN + h;
            if (row >= 0 && row < valid_rows) {
                output[out_idx] = table[row * TILE_HIDDEN + h];
            } else {
                output[out_idx] = 0.0f;
            }
        }
    }

    event1();
}

}