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

#include "dali/kernels/signal/decibel/to_decibels_cpu.h"
#include <cmath>
#include <complex>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
ToDecibelsCpu<T>::~ToDecibelsCpu() = default;

template <typename T>
KernelRequirements ToDecibelsCpu<T>::Setup(
    KernelContext &context,
    const InTensorCPU<T, DynamicDimensions> &in,
    const ToDecibelsArgs<T> &args) {
  auto out_shape = in.shape;
  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  KernelRequirements req;
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};
  return req;
}

template <typename T>
void ToDecibelsCpu<T>::Run(
    KernelContext &context,
    const OutTensorCPU<T, DynamicDimensions> &out,
    const InTensorCPU<T, DynamicDimensions> &in,
    const ToDecibelsArgs<T> &args) {
  auto in_size = volume(in.shape);
  auto out_size = volume(out.shape);
  assert(out_size == in_size);

  auto s_ref = args.s_ref;
  if (args.ref_max) {
    s_ref = 0.0;
    for (int64_t i = 0; i < in_size; i++) {
      if (in.data[i] > s_ref)
        s_ref = in.data[i];
    }
    // avoid division by 0
    if (s_ref == 0.0)
      s_ref = 1.0;
  }

  MagnitudeToDecibel<T> dB(args.multiplier, s_ref, args.min_ratio);
  for (int64_t i = 0; i < in_size; i++) {
    out.data[i] = dB(in.data[i]);
  }
}

template class ToDecibelsCpu<float>;
template class ToDecibelsCpu<double>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
