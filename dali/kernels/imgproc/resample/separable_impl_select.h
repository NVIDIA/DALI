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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_SELECT_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_SELECT_H_

#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/imgproc/resample/separable_impl.h"

namespace dali {
namespace kernels {

template <typename OutputElement, typename InputElement>
typename SeparableResamplingFilter<OutputElement, InputElement>::Ptr
SeparableResamplingFilter<OutputElement, InputElement>::Create(const Params &params) {
  (void)params;
  using ImplType = SeparableResamplingGPUImpl<OutputElement, InputElement>;
  return Ptr(new ImplType());
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_SELECT_H_
