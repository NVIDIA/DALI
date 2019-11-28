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

#include "dali/kernels/signal/decibel/amplitude_to_db_cpu.h"
#include <cmath>
#include <complex>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T, int Dims>
AmplitudeToDbCpu<T, Dims>::~AmplitudeToDbCpu() = default;

template <typename T, int Dims>
KernelRequirements AmplitudeToDbCpu<T, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<T, Dims> &in,
    const AmplitudeToDbArgs<T> &args) {
  auto out_shape = in.shape;
  auto data_size = volume(in.shape);
  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  KernelRequirements req;
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};

  s_ref_ = args.s_ref;
  if (args.ref_max) {
    s_ref_ = 0.0;
    for (int64_t i = 0; i < data_size; i++) {
      if (in.data[i] > s_ref_)
        s_ref_ = in.data[i];
    }
  }
  return req;
}

template <typename T, int Dims>
void AmplitudeToDbCpu<T, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<T, Dims> &out,
    const InTensorCPU<T, Dims> &in,
    const AmplitudeToDbArgs<T> &args) {
  auto in_size = volume(in.shape);
  auto out_size = volume(out.shape);
  assert(out_size == in_size);

  DecibelCalculator<T> dB(args.multiplier, s_ref_, args.min_ratio);
  for (int64_t i = 0; i < in_size; i++) {
    out.data[i] = dB(in.data[i]);
  }
}

template class AmplitudeToDbCpu<float, 1>;
template class AmplitudeToDbCpu<float, 2>;
template class AmplitudeToDbCpu<float, 3>;
template class AmplitudeToDbCpu<float, 4>;

template class AmplitudeToDbCpu<double, 1>;
template class AmplitudeToDbCpu<double, 2>;
template class AmplitudeToDbCpu<double, 3>;
template class AmplitudeToDbCpu<double, 4>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
