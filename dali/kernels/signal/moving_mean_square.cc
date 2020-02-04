// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/**
 * Accurate type is such type, that doesn't require
 * reset interval for maintaining numeric precision
 */
template<typename T>
struct needs_reset {
  static constexpr bool value = !(std::is_integral<T>::value && sizeof(T) <= 2);
};


/**
 * Type of the accumulator:
 * In case calculation is performed on floating-point,
 * it requires reset_interval to keep numeric accuracy.
 */
template<typename T>
struct accumulator_type {
  using type = std::conditional_t<needs_reset<T>::value, float, int64_t>;
};

template<typename T>
using acc_t = typename accumulator_type<T>::type;


template<typename T>
acc_t<T> Square(const T &val) {
  acc_t<T> res = val;
  return res * res;
}


template<typename T>
acc_t<T> CalcSumSquared(span<const T> values) {
  acc_t<T> sumsq = 0;
  for (const auto &val : values) {
    sumsq += Square(val);
  }
  return sumsq;
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
  TensorShape<> out_shape = {in.shape[0] - args.window_size + 1};
  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};
  return req;
}


template<typename T>
void CalcMovingMeanSquare(span<float> out, span<const T> in, int length, float mean_factor,
                          int window_size, int reset_interval = -1) {
  reset_interval = reset_interval == -1 ? length : reset_interval;
  acc_t<T> sumsq = 0;
  for (int window_begin = 0; window_begin <= length - window_size;) {
    sumsq = CalcSumSquared(make_span(&in[window_begin], window_size));
    out[window_begin] = sumsq * mean_factor;
    auto interval_end = std::min(window_begin + reset_interval, length) - window_size + 1;
    for (window_begin++; window_begin < interval_end; window_begin++) {
      sumsq += Square(in[window_begin + window_size - 1]) - Square(in[window_begin - 1]);
      out[window_begin] = sumsq * mean_factor;
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
