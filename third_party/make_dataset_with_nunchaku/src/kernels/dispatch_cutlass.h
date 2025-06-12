#pragma once

#include "Tensor.h"
#include "common.h"

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

template <typename F>
inline void dispatchF16(Tensor::ScalarType type, F &&func) {
  if (type == Tensor::FP16) {
    func.template operator()<cutlass::half_t>();
  } else if (type == Tensor::BF16) {
    func.template operator()<cutlass::bfloat16_t>();
  } else {
    assert(false);
  }
}
