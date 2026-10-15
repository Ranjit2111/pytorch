#include <ATen/native/cuda/Gfx942Tf32Rounding.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include <c10/util/irange.h>

#include <cstdint>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/as_strided.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/expand.h>
#endif

namespace at::native {

Tensor round_gfx942_tf32_rne_cuda(const Tensor& input) {
  TORCH_INTERNAL_ASSERT(input.scalar_type() == ScalarType::Float);

  // Expanded dimensions contain repeated values, so round their size-one
  // source instead of materializing every broadcasted element. Expanding the
  // result restores the original logical shape and zero strides.
  auto rounding_sizes = input.sizes().vec();
  bool has_expanded_dimension = false;
  for (const auto dim : c10::irange(input.dim())) {
    if (rounding_sizes[dim] > 1 && input.stride(dim) == 0) {
      rounding_sizes[dim] = 1;
      has_expanded_dimension = true;
    }
  }

  Tensor rounding_input = input;
  if (has_expanded_dimension) {
    rounding_input = input.as_strided(rounding_sizes, input.strides());
  }

  auto output =
      at::empty_like(rounding_input, at::MemoryFormat::Preserve);
  auto iter = TensorIteratorConfig()
                  .add_output(output)
                  .add_const_input(rounding_input)
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .build();

  gpu_kernel(iter, [] GPU_LAMBDA(float value) -> float {
    union {
      float fp32;
      uint32_t bits;
    } converted = {value};

    if (~converted.bits & 0x7f800000U) {
      converted.bits += 0xfffU + ((converted.bits >> 13) & 1U);
    } else if (converted.bits & 0x1fffU) {
      converted.bits |= 0x2000U;
    }
    converted.bits &= 0xffffe000U;
    return converted.fp32;
  });
  return has_expanded_dimension ? output.expand(input.sizes()) : output;
}

} // namespace at::native
