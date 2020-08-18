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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_H_

#include <cuda_runtime.h>
#include <memory>
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/imgproc/resample/separable.h"

namespace dali {
namespace kernels {

template <typename OutputElement, typename InputElement, int _spatial_ndim = 2>
struct ResampleGPU {
  static_assert(_spatial_ndim == 2 || _spatial_ndim == 3, "Only 2D and 3D resampling is supported");
  static constexpr int spatial_ndim = _spatial_ndim;
  static constexpr int ndim = spatial_ndim + 1;
  using Input = InListGPU<InputElement, ndim>;
  using Output = OutListGPU<OutputElement, ndim>;

  using Impl = SeparableResamplingFilter<OutputElement, InputElement, spatial_ndim>;
  using Params = typename Impl::Params;
  using ImplPtr = typename Impl::Ptr;

  ImplPtr pImpl;

  Impl *SelectImpl(
      KernelContext &context,
      const Input &input,
      const Params &params) {
    if (!pImpl)
      pImpl = Impl::Create(params);
    return pImpl.get();
  }

  KernelRequirements Setup(KernelContext &context, const Input &input, const Params &params) {
    auto *impl = SelectImpl(context, input, params);
    return impl->Setup(context, input, params);
  }

  void Run(KernelContext &context, const Output &output, const Input &input, const Params &params) {
    assert(pImpl != nullptr);
    pImpl->Run(context, output, input, params);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_H_
