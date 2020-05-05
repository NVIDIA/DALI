// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_REDUCE_REDUCE_GPU_H_
#define DALI_KERNELS_REDUCE_REDUCE_GPU_H_

#include <memory>
#include "dali/kernels/kernel.h"
#include "dali/core/tensor_view.h"

namespace dali {
namespace kernels {

template <typename Out, typename In>
class DLL_PUBLIC SumGPU {
 public:
  SumGPU();
  ~SumGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class SumGPU<uint64_t, uint8_t>;
extern template class SumGPU<float, uint8_t>;
extern template class SumGPU<int64_t, int8_t>;
extern template class SumGPU<float, int8_t>;
extern template class SumGPU<uint64_t, uint16_t>;
extern template class SumGPU<float, uint16_t>;
extern template class SumGPU<int64_t, int16_t>;
extern template class SumGPU<float, int16_t>;
extern template class SumGPU<int64_t, int32_t>;
extern template class SumGPU<float, int32_t>;
extern template class SumGPU<float, float>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_GPU_H_
