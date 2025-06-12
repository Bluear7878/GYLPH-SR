#pragma once

#include "Tensor.h"
#include "common.h"

// Tensor depthwise_conv2d_kernel(Tensor A, Tensor B);

Tensor dwconv_f16(Tensor input, Tensor weight, Tensor out, Tensor bias);
