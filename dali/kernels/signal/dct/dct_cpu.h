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

#ifndef DALI_KERNELS_SIGNAL_DCT_DCT_CPU_H_
#define DALI_KERNELS_SIGNAL_DCT_DCT_CPU_H_

#include <memory>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/dct/dct_args.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

/**
 * @brief Discrete Cosine Transform 1D CPU kernel.
 *        Performs a DCT transformation over a single dimension in a multi-dimensional input.
 *
 * @remarks It supports DCT types I, II, III and IV decribed here:
 *          https://en.wikipedia.org/wiki/Discrete_cosine_transform
 *          DCT generally stands for type II and inverse DCT stands for DCT type III
 *
 * @see DCTArgs
 */
template <typename OutputType = float,  typename InputType = float, int Dims = 2>
class DLL_PUBLIC Dct1DCpu {
 public:
  static_assert(std::is_floating_point<InputType>::value,
    "Data type should be floating point");
  static_assert(std::is_same<OutputType, InputType>::value,
    "Data type conversion is not supported");

  DLL_PUBLIC ~Dct1DCpu();

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, Dims> &in,
                                      const DctArgs &args, int axis);

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutTensorCPU<OutputType, Dims> &out,
                      const InTensorCPU<InputType, Dims> &in,
                      const DctArgs &args, int axis);
 private:
  std::vector<OutputType> cos_table_;
  DctArgs args_;
  int axis_;
};

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DCT_DCT_CPU_H_
