// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

namespace dali {
namespace kernels {
namespace signal {

namespace {

template<typename T>
acc_t<T> Square(const T &val) {
  acc_t<T> res = val;
  return res * res;
}

}  // namespace

template<typename T>
MovingMeanSquareCpu<T>::~MovingMeanSquareCpu() = default;


template<typename T>
KernelRequirements
MovingMeanSquareCpu<T>::Setup(KernelContext &context, const InTensorCPU<T, 1> &in,
                              const MovingMeanSquareArgs &args) {
  DALI_ENFORCE(args.window_size <= in.num_elements(),
               make_string("window_size can't be bigger than input buffer. Received: window_size=",
                           args.window_size, ", input_size=", in.num_elements()));
  KernelRequirements req;
  req.output_shapes.push_back(uniform_list_shape<>(1, in.shape));
  return req;
}


template<typename T>
void CalcMovingMeanSquare(span<float> out, span<const T> in, int length, float mean_factor,
                          int window_size, int reset_interval = -1) {
  reset_interval = reset_interval == -1 ? length : reset_interval;

  assert(out.size() == in.size());
  assert(out.size() == length);
  for (int64_t out_pos = 0; out_pos < length; out_pos++) {
    acc_t<T> sumsq = 0;
    int64_t win_begin = out_pos - window_size + 1;
    for (int64_t pos = std::max<int64_t>(win_begin, 0); pos <= out_pos; pos++) {
      sumsq += Square(in[pos]);
    }
    out[out_pos] = sumsq * mean_factor;
    int64_t interval_end = std::min<int64_t>(length, out_pos + reset_interval);
    for ( ; out_pos < interval_end; ) {
      out_pos++;
      win_begin++;
      sumsq += Square(in[out_pos]);
      if (win_begin > 0)
        sumsq -= Square(in[win_begin - 1]);
      out[out_pos] = sumsq * mean_factor;
    }
  }
}


template<typename T>
void MovingMeanSquareCpu<T>::Run(KernelContext &context, const OutTensorCPU<float, 1> &out,
                                 const InTensorCPU<T, 1> &in, const MovingMeanSquareArgs &args) {
  const auto length = in.shape[0];
  auto sp_in = make_cspan(in.data, in.shape[0]);
  auto sp_out = make_span(out.data, out.shape[0]);
  const float mean_factor = 1.f / args.window_size;
  const int reset_interval = needs_reset<T>::value ? args.reset_interval : -1;

  CalcMovingMeanSquare(sp_out, sp_in, length, mean_factor, args.window_size, reset_interval);
}


template class MovingMeanSquareCpu<double>;
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
