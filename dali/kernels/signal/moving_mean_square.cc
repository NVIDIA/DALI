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

template<typename Result = float, typename T>
float Square(const T &val) {
  Result res = val;
  return res * res;
}


template<typename T>
float CalcSumSquared(span<const T> buffer, int start, int length) {
  DALI_ENFORCE(buffer.size() >= length + start,
               make_string_delim(" ", "Buffer overflow (size:", buffer.size(), "length:", length,
                                 "start:", start));
  float sumsq = 0;
  for (int i = start; i < length + start; i++) {
    sumsq += Square(buffer[i]);
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
  KernelRequirements req;
  req.output_shapes = {TensorListShape<>({in.shape})};
  return req;
}


template<typename T>
void with_modulo(span<const T> spin, span<float> spout, int length, float mean_factor,
                 int reset_interval, int window_size) {
  float sumsq;
  for (int window_begin = 0; window_begin <= length - window_size; window_begin++) {
    if (window_begin % reset_interval == 0) {
      sumsq = CalcSumSquared(spin, window_begin, window_size);
    } else {
      sumsq += Square(spin[window_begin + window_size - 1]) - Square(spin[window_begin - 1]);
    }
    spout[window_begin] = sumsq * mean_factor;
  }
  for (int i = length - window_size + 1; i < length; i++) {
    if (i % reset_interval == 0) {
      sumsq = CalcSumSquared(spin, i, length - i);
    } else {
      sumsq -= Square(spin[i - 1]);
    }
    spout[i] = sumsq * mean_factor;
  }
}


template<typename T>
void
with_loop(span<const T> in, span<float> out, int length, float mean_factor, int reset_interval,
          int window_size) {
  float sumsq = 0;
  int cnt = 0;
  int window_begin = 0;
  bool recalc = true;
  while (window_begin <= length - window_size) {
    if (recalc) {
      sumsq = CalcSumSquared(in, window_begin, window_size); // krotsze dane niz window size
      out[window_begin++] = sumsq * mean_factor;
      recalc = false;
      cnt++;
    }
    while (window_begin < reset_interval * cnt && window_begin <= length - window_size) {
      sumsq += Square(in[window_begin + window_size - 1]) - Square(in[window_begin - 1]);
      out[window_begin++] = sumsq * mean_factor;
    }
    if (window_begin >= reset_interval * cnt) {
      recalc = true;
    }
  }
  while (window_begin < length) {
    if (recalc) {
      sumsq = CalcSumSquared(in, window_begin, length - window_begin);
      out[window_begin++] = sumsq * mean_factor;
      recalc = false;
      cnt++;
    }
    while (window_begin < reset_interval * cnt && window_begin < length) {
      sumsq -= Square(in[window_begin - 1]);
      out[window_begin++] = sumsq * mean_factor;
    }
    if (window_begin >= reset_interval * cnt) {
      recalc = true;
    }
  }
}


template<typename T>
void MovingMeanSquareCpu<T>::Run(KernelContext &context, const OutTensorCPU<float, 1> &out,
                                 const InTensorCPU<T, 1> &in, const MovingMeanSquareArgs &args) {
  const auto length = in.shape[0];
  auto spin = make_cspan(in.data, length);
  auto spout = make_span(out.data, length);
  const float mean_factor = 1.f / args.window_size;
  const int reset_interval = args.reset_interval == -1 ? length : args.reset_interval;

//  using namespace std::chrono;  // NOLINT
//  using hrc = std::chrono::high_resolution_clock;
//  auto start = hrc::now();
  with_modulo(spin, spout, length, mean_factor, reset_interval, args.window_size);
//  with_loop(spin, spout, length, mean_factor, reset_interval, args.window_size);
//  auto stop = hrc::now();
//  cout << duration_cast<microseconds>(stop - start).count() << "\n";
}


template
class MovingMeanSquareCpu<float>;

template
class MovingMeanSquareCpu<uint8_t>;

template
class MovingMeanSquareCpu<int8_t>;

template
class MovingMeanSquareCpu<uint16_t>;

template
class MovingMeanSquareCpu<int16_t>;

template
class MovingMeanSquareCpu<uint32_t>;

template
class MovingMeanSquareCpu<int32_t>;

template
class MovingMeanSquareCpu<uint64_t>;

template
class MovingMeanSquareCpu<int64_t>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
