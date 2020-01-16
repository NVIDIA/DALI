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

namespace {
template<typename T>
float Power2(const T &val) {
  return static_cast<float>(val) * val;
}
}

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
  const auto length = in.shape[0];
  auto spin = make_cspan(in.data, length);
  auto spout = make_span(out.data, length);
  const float mean_factor = 1.f / args.window_size;
  const int reset_interval = args.reset_interval == -1 ? length : args.reset_interval;

  float sumsq;
  for (int window_begin = 0; window_begin <= length - args.window_size; window_begin++) {
    if (window_begin % reset_interval == 0) {
      sumsq = detail::CalcSumSquared(spin, window_begin, args.window_size);
    } else {
      sumsq += Power2(spin[window_begin + args.window_size - 1]) - Power2(spin[window_begin - 1]);
    }
    spout[window_begin] = sumsq * mean_factor;
  }
  for (int i = length - args.window_size + 1; i < length; i++) {
    if (i % reset_interval == 0) {
      sumsq = detail::CalcSumSquared(spin, i, length - i);
    } else {
      sumsq -= Power2(spin[i - 1]);
    }
    spout[i] = sumsq * mean_factor;
  }
}

template class MovingMeanSquareCpu<float>;
template class MovingMeanSquareCpu<uint8_t>;
template class MovingMeanSquareCpu<int8_t>;
template class MovingMeanSquareCpu<uint16_t>;
template class MovingMeanSquareCpu<int16_t>;
template class MovingMeanSquareCpu<uint32_t>;
template class MovingMeanSquareCpu<int32_t>;
template class MovingMeanSquareCpu<uint64_t>;
template class MovingMeanSquareCpu<int64_t>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
