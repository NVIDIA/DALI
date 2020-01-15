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

#include "dali/kernels/signal/moving_mean_square.h"

namespace dali {
namespace kernels {
namespace signal {


template<typename T>
MovingMeanSquareCpu<T>::~MovingMeanSquareCpu() = default;


template<typename T>
KernelRequirements
MovingMeanSquareCpu<T>::Setup(KernelContext &context, const InTensorCPU<T, 1> &in,
                              const MovingMeanSquareArgs &args) {
  KernelRequirements req;
  req.output_shapes = {TensorListShape<>({in.shape})};
  return req;
}


template<typename T>
void MovingMeanSquareCpu<T>::Run(KernelContext &context, const OutTensorCPU<float, 1> &out,
                                 const InTensorCPU<T, 1> &in, const MovingMeanSquareArgs &args) {
  const auto in_ptr = in.data;
  const auto out_ptr = out.data;
  const auto length = in.shape[0];
  const float mean_factor = 1.f / args.window_size;
  T sumsq = 0;
  for (int i = 0; i < args.window_size; i++) {
    sumsq += in_ptr[i] * in_ptr[i];
  }
  for (int i = 0; i < length - args.window_size; i++) {
    out_ptr[i] = sumsq * mean_factor;
    sumsq -= in_ptr[i] * in_ptr[i];
    sumsq += in_ptr[i + args.window_size] * in_ptr[i + args.window_size];
  }
  for (int i = length - args.window_size; i < length; i++) {
    out_ptr[i] = sumsq * mean_factor;
    sumsq -= in_ptr[i] * in_ptr[i];
  }
}


template class MovingMeanSquareCpu<float>;
template class MovingMeanSquareCpu<int>;
//template class ToDecibelsCpu<float, 3>;
//template class ToDecibelsCpu<float, 4>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
