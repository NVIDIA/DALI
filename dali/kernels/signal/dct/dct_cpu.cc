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

#include "dali/kernels/signal/dct/dct_cpu.h"
#include <cmath>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

template <typename OutputType, typename InputType, int Dims>
Dct1DCpu<OutputType, InputType, Dims>::~Dct1DCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements Dct1DCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, Dims> &in,
    const DctArgs &args) {
  KernelRequirements req;
  return req;
}

template <typename OutputType, typename InputType, int Dims>
void Dct1DCpu<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<OutputType, Dims> &out,
    const InTensorCPU<InputType, Dims> &in,
    const DctArgs &args) {
}

template class Dct1DCpu<float, float, 1>;
template class Dct1DCpu<float, float, 2>;
template class Dct1DCpu<float, float, 3>;

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
