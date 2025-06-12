#pragma once

#include "Tensor.h"
#include "common.h"

Tensor gemm_batched_fp16(Tensor a,  // FP16 row-major [(... batch ...), M, K]
                         Tensor b,  // FP16 col-major [(... batch ...), N, K]
                         Tensor out // FP32 row-major [(... batch ...), M, N]
);
