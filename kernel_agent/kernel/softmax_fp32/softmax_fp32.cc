#define NOCPP

#include <aie_api/aie.hpp>

extern "C" {

static inline float fast_exp(float x) {
    if (x < -20.0f) {
        return 0.0f;
    }
    float t = 1.0f + x * 0.00390625f;
    if (t < 0.0f) {
        t = 0.0f;
    }
    t = t * t;
    t = t * t;
    t = t * t;
    t = t * t;
    t = t * t;
    t = t * t;
    t = t * t;
    t = t * t;
    return t;
}

void softmax_fp32(float input[113], float output[113]) {
    constexpr int N = 113;

    event0();

    float maxv = input[0];
    for (int i = 1; i < N; ++i) {
        if (input[i] > maxv) {
            maxv = input[i];
        }
    }

    float sum = 0.0f;
    float tmp[N];
    for (int i = 0; i < N; ++i) {
        const float z = input[i] - maxv;
        float e = fast_exp(z);
        tmp[i] = e;
        sum += e;
    }

    float inv_sum = 0.0f;
    if (sum > 0.0f) {
        inv_sum = 1.0f / sum;
    }

    for (int i = 0; i < N; ++i) {
        output[i] = tmp[i] * inv_sum;
    }

    event1();
}

}
