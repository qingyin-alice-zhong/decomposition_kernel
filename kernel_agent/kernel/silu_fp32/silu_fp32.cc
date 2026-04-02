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

void silu_fp32(float input[720], float output[720]) {
    constexpr int N = 720;

    event0();

    for (int i = 0; i < N; ++i) {
        const float x = input[i];
        float sigmoid;
        if (x >= 0.0f) {
            const float z = fast_exp(-x);
            sigmoid = 1.0f / (1.0f + z);
        } else {
            const float z = fast_exp(x);
            sigmoid = z / (1.0f + z);
        }
        output[i] = x * sigmoid;
    }

    event1();
}

}
