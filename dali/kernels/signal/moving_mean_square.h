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

#ifndef DALI_KERdsfasdfasdfS_CPU_H_
#define DALI_KERdsfasdfasdfS_CPU_H_

//#include <memory>
//#include "dali/core/common.h"
//#include "dali/core/error_handling.h"
//#include "dali/core/format.h"
//#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/moving_mean_square_args.h"

namespace dali {
namespace kernels {
namespace signal {

template<typename T = float>
class DLL_PUBLIC MovingMeanSquareCpu {
 public:

  DLL_PUBLIC ~MovingMeanSquareCpu();

  DLL_PUBLIC KernelRequirements
  Setup(KernelContext &context, const InTensorCPU<T, 1> &in, const MovingMeanSquareArgs &args);

  DLL_PUBLIC void
  Run(KernelContext &context, const OutTensorCPU<float, 1> &out, const InTensorCPU<T, 1> &in,
      const MovingMeanSquareArgs &args);
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_CPU_H_
