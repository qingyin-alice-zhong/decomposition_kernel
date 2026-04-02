#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

void layer_norm_768_fp32(float input[768], float output[768], float param[1538]) {
    constexpr int HIDDEN = 768;
    constexpr float INV_HIDDEN = 1.0f / 768.0f;

    event0();

    const float* gamma = param;
    const float* beta = param + HIDDEN;
    const float eps = param[2 * HIDDEN];

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = 0; i < HIDDEN; ++i) {
        const float x = input[i];
        sum += x;
        sumsq += x * x;
    }

    const float mean = sum * INV_HIDDEN;
    float var = sumsq * INV_HIDDEN - mean * mean;
    if (var < 0.0f) {
        var = 0.0f;
    }

    float s = var + eps;
    if (s <= 0.0f) {
        s = eps;
    }

    union {
        float f;
        uint32_t i;
    } conv;
    conv.f = s;
    conv.i = 0x5f3759df - (conv.i >> 1);
    float inv_std = conv.f;
    inv_std = inv_std * (1.5f - 0.5f * s * inv_std * inv_std);
    inv_std = inv_std * (1.5f - 0.5f * s * inv_std * inv_std);

    for (int i = 0; i < HIDDEN; ++i) {
        const float norm = (input[i] - mean) * inv_std;
        output[i] = norm * gamma[i] + beta[i];
    }

    event1();
}

}
