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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_CPU_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_CPU_H_

#include <stdexcept>
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/imgproc/resample/separable_cpu.h"

namespace dali {
namespace kernels {

template <typename OutputElement, typename InputElement>
struct ResampleCPU {
  using Input =  InTensorCPU<InputElement, 3>;
  using Output = OutTensorCPU<OutputElement, 3>;

  using Impl = SeparableResampleCPU<OutputElement, InputElement>;

  static Impl *GetImpl(KernelContext &context, bool create = false) {
    Impl *impl = any_cast<Impl>(&context.kernel_data);
    if (!impl) {
      if (!create)
        throw std::logic_error("The context is not valid!\n"
                               "Hint: have you called GetRequirements using this context?");

      context.kernel_data = Impl();
      impl = any_cast<Impl>(&context.kernel_data);
    }
    return impl;
  }

  static KernelRequirements GetRequirements(KernelContext &context,
                                            const Input &input,
                                            const ResamplingParams2D &params) {
    auto *impl = GetImpl(context, true);
    return impl->Setup(context, input, params);
  }

  static void Run(KernelContext &context,
                  const Output &output,
                  const Input &input,
                  const ResamplingParams2D &params) {
    Impl *impl = GetImpl(context);
    impl->Run(context, output, input, params);
  }
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_CPU_H_
