//===- add.cc -------------------------------------------------*- C++ -*-===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void eltwise_add_int16_vector(int16_t a_in[1024], int16_t b_in[1024], int16_t c_out[1024]) {
  constexpr int vec_factor = 16;
  event0();
  int16_t *__restrict pA1 = a_in;
  int16_t *__restrict pB1 = b_in;
  int16_t *__restrict pC1 = c_out;
  const int F = 1024 / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<int16_t, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::vector<int16_t, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
      pB1 += vec_factor;
      aie::vector<int16_t, vec_factor> cout = aie::add(A0, B0);
      aie::store_v(pC1, cout);
      pC1 += vec_factor;
    }
  event1();
}

} // extern "C"
