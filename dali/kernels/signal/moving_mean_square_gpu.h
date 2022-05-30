// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_GPU_H_
#define DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_GPU_H_

#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/moving_mean_square_args.h"

namespace dali {
namespace kernels {
namespace signal {

template<typename InputType>
class DLL_PUBLIC MovingMeanSquareGpu {
 public:
  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InListGPU<InputType, 1> &in);

  DLL_PUBLIC void Run(KernelContext &context, const OutListGPU<float, 1> &out,
                      const InListGPU<InputType, 1> &in, const MovingMeanSquareArgs &args);
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_GPU_H_
