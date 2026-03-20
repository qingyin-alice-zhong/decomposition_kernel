//===- add.cc -------------------------------------------------*- C++ -*-===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void eltwise_add_int16_vector(int16_t a_in[1024], int16_t b_in[1024], int16_t c_out[1024]) {
    // TODO: Implement the kernel
}

} // extern "C"