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

template <typename OutputElement, typename InputElement>
struct ResampleGPU {
  using Input = InListGPU<InputElement, 3>;
  using Output = OutListGPU<OutputElement, 3>;

  using Impl = SeparableResamplingFilter<OutputElement, InputElement>;
  using Params = typename Impl::Params;
  using ImplPtr = typename Impl::Ptr;

  static Impl *SelectImpl(
      KernelContext &context,
      const Input &input,
      const Params &params) {
    auto cur_impl = any_cast<ImplPtr>(&context.kernel_data);
    if (cur_impl) {
      return cur_impl->get();
    } else {
      auto ptr = Impl::Create(params);
      context.kernel_data = ptr;
      return ptr.get();
    }
  }
  static Impl *GetImpl(KernelContext &context) {
    auto cur_impl = any_cast<ImplPtr>(&context.kernel_data);
    if (cur_impl) {
      return cur_impl->get();
    } else {
      return nullptr;
    }
  }

  static KernelRequirements
  GetRequirements(KernelContext &context, const Input &input, const Params &params) {
    auto *impl = SelectImpl(context, input, params);
    return impl->Setup(context, input, params);
  }

  static void
  Run(KernelContext &context, const Output &output, const Input &input, const Params &params) {
    Impl *impl = GetImpl(context);
    assert(impl != nullptr);
    impl->Run(context, output, input, params);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_H_
