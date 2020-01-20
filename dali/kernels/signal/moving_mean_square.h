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

#ifndef DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_H_
#define DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_H_

#include "dali/kernels/kernel.h"
#include "dali/core/format.h"
#include "dali/kernels/signal/moving_mean_square_args.h"

namespace dali {
namespace kernels {
namespace signal {

template<typename InputType>
class DLL_PUBLIC MovingMeanSquareCpu {
 public:
  DLL_PUBLIC ~MovingMeanSquareCpu();

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InTensorCPU<InputType, 1> &in,
                                      const MovingMeanSquareArgs &args);

  DLL_PUBLIC void Run(KernelContext &context, const OutTensorCPU<float, 1> &out,
                      const InTensorCPU<InputType, 1> &in, const MovingMeanSquareArgs &args);
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_H_
