#pragma once

#include "Tensor.h"
#include "common.h"

Tensor awq_gemm_forward_cuda(Tensor _in_feats, Tensor _kernel, Tensor _scales,
                             Tensor _zeros);
