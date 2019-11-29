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

#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu.h"
#include <cmath>
#include <complex>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu_impl.h"

namespace dali {
namespace kernels {
namespace audio {

template <typename T, int Dims>
MelFilterBankCpu<T, Dims>::~MelFilterBankCpu() = default;

template <typename T, int Dims>
KernelRequirements MelFilterBankCpu<T, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<T, Dims> &in,
    const MelFilterBankArgs &args) {
  auto out_shape = in.shape;
  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  KernelRequirements req;
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};
  return req;
}

template <typename T, int Dims>
void MelFilterBankCpu<T, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<T, Dims> &out,
    const InTensorCPU<T, Dims> &in,
    const MelFilterBankArgs &args) {

      // TODO
}

template class MelFilterBankCpu<float, 2>;
template class MelFilterBankCpu<double, 2>;

}  // namespace audio
}  // namespace kernels
}  // namespace dali
