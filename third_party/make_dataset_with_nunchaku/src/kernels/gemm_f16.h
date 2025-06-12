#pragma once

#include "Tensor.h"
#include "common.h"

Tensor gemm_f16(Tensor input,  // FP16
                Tensor weight, // FP16
                Tensor out,    // FP16
                Tensor bias, float alpha);
