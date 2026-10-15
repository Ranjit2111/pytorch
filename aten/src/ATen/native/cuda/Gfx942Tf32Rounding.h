#pragma once

#include <ATen/core/Tensor.h>

namespace at::native {

Tensor round_gfx942_tf32_rne_cuda(const Tensor& input);

} // namespace at::native
