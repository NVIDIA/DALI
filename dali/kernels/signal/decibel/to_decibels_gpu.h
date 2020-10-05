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

#ifndef DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_GPU_H_
#define DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_GPU_H_

#include <memory>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/decibel/to_decibels_args.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T = float>
class DLL_PUBLIC ToDecibelsGpu {
 public:
  static_assert(std::is_floating_point<T>::value,
    "Only floating point types are supported");

  DLL_PUBLIC ~ToDecibelsGpu();

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<T, DynamicDimensions> &in);

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutListGPU<T, DynamicDimensions> &out,
                      const InListGPU<T, DynamicDimensions> &in,
                      const ToDecibelsArgs<T> &args,
                      InListGPU<T, 0> max_values = {});
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_GPU_H_
