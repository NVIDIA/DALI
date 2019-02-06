// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_H_

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {
namespace kernels {


template <typename OutputElement, typename InputElement>
struct SeparableResamplingFilter {
  using Input = InListGPU<InputElement, 3>;
  using Output = OutListGPU<OutputElement, 3>;

  virtual ~SeparableResamplingFilter() = default;

  using Params = std::vector<ResamplingParams2D>;

  virtual KernelRequirements Setup(KernelContext &context, const Input &in, const Params &params) = 0;
  virtual void Run(KernelContext &context, const Output &out, const Input &in, const Params &params) = 0;
  using Ptr = std::unique_ptr<SeparableResamplingFilter>;

  static Ptr Create(const Params &params);
};

}  // namespace kernels
}  // namespace dali

#include "separable_impl_select.h"

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_H_
