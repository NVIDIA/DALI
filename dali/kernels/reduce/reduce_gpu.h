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
class SumGPU {
 public:
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes);

  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_GPU_H_
