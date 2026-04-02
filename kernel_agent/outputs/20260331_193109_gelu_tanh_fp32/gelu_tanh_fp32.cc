#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void gelu_tanh_fp32(float input[1024], float output[1024]) {
    constexpr int TILE_ELEMS = 1024;
    constexpr float INV_SQRT_2PI = 0.7978845608f;
    constexpr float GELU_COEFF = 0.044715f;

    event0();

    for (int i = 0; i < TILE_ELEMS; ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        float u = INV_SQRT_2PI * (x + GELU_COEFF * x3);
        if (u > 3.0f) {
            u = 3.0f;
        } else if (u < -3.0f) {
            u = -3.0f;
        }
        const float u2 = u * u;
        const float tanh_approx = u * (27.0f + u2) / (27.0f + 9.0f * u2);
        output[i] = 0.5f * x * (1.0f + tanh_approx);
    }

    event1();
}

}
