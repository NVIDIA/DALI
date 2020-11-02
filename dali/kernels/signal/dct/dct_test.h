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

#ifndef DALI_KERNELS_SIGNAL_DCT_DCT_TEST_H_
#define DALI_KERNELS_SIGNAL_DCT_DCT_TEST_H_

#include <cmath>

namespace dali {
namespace kernels {
namespace signal {
namespace dct {
namespace test {

template <typename T>
void ReferenceDctTypeI(span<T> out, span<const T> in, bool normalize, float lifter) {
  int64_t in_length = in.size();
  int64_t out_length = out.size();
  double phase_mul = M_PI / (in_length - 1);
  for (int64_t k = 0; k < out_length; k++) {
    double sign = (k % 2 == 0) ? 1 : -1;
    double out_val = 0.5 * (in[0] + sign * in[in_length-1]);
    for (int64_t n = 1; n < in_length - 1; n++) {
      out_val += in[n] * std::cos(phase_mul * n * k);
    }
    float coeff = lifter ? (1.0 + lifter / 2 * std::sin(M_PI / lifter * (k + 1))) : 1.f;
    out[k] = out_val * coeff;
  }
}

template <typename T>
void ReferenceDctTypeII(span<T> out, span<const T> in, bool normalize, float lifter) {
  int64_t in_length = in.size();
  int64_t out_length = out.size();
  double phase_mul = M_PI / in_length;
  double factor_k_0 = 1, factor_k_i = 1;
  if (normalize) {
    factor_k_i = std::sqrt(2.0 / in_length);
    factor_k_0 = 1.0 / std::sqrt(in_length);
  }
  for (int64_t k = 0; k < out_length; k++) {
    double out_val = 0;
    for (int64_t n = 0; n < in_length; n++) {
      out_val += in[n] * std::cos(phase_mul * (n + 0.5) * k);
    }
    double factor = (k == 0) ? factor_k_0 : factor_k_i;
    float coeff = lifter ? (1.0 + lifter / 2 * std::sin(M_PI / lifter * (k + 1))) : 1.f;
    out[k] = factor * out_val * coeff;
  }
}

template <typename T>
void ReferenceDctTypeIII(span<T> out, span<const T> in, bool normalize, float lifter) {
  int64_t in_length = in.size();
  int64_t out_length = out.size();
  double phase_mul = M_PI / in_length;
  double factor_n_0 = 0.5, factor_n_i = 1;
  if (normalize) {
    factor_n_i = std::sqrt(2.0 / in_length);
    factor_n_0 = 1.0 / std::sqrt(in_length);
  }

  for (int64_t k = 0; k < out_length; k++) {
    double out_val = factor_n_0 * in[0];
    for (int64_t n = 1; n < in_length; n++) {
      out_val += factor_n_i * in[n] * std::cos(phase_mul * n * (k + 0.5));
    }
    float coeff = lifter ? (1.0 + lifter / 2 * std::sin(M_PI / lifter * (k + 1))) : 1.f;
    out[k] = out_val * coeff;
  }
}

template <typename T>
void ReferenceDctTypeIV(span<T> out, span<const T> in, bool normalize, float lifter) {
  int64_t in_length = in.size();
  int64_t out_length = out.size();
  double phase_mul = M_PI / in_length;
  double factor = normalize ? std::sqrt(2.0 / in_length) : 1.0;
  for (int64_t k = 0; k < out_length; k++) {
    double out_val = 0;
    for (int64_t n = 0; n < in_length; n++) {
      out_val += factor * in[n] * std::cos(phase_mul * (n + 0.5) * (k + 0.5));
    }
    float coeff = lifter ? (1.0 + lifter / 2 * std::sin(M_PI / lifter * (k + 1))) : 1.f;
    out[k] = out_val * coeff;
  }
}


template <typename T>
void ReferenceDct(int dct_type, span<T> out, span<const T> in, bool normalize, float lifter = 0) {
  switch (dct_type) {
    case 1:
      ReferenceDctTypeI(out, in, normalize, lifter);
      break;

    case 2:
      ReferenceDctTypeII(out, in, normalize, lifter);
      break;

    case 3:
      ReferenceDctTypeIII(out, in, normalize, lifter);
      break;

    case 4:
      ReferenceDctTypeIV(out, in, normalize, lifter);
      break;

    default:
      ASSERT_TRUE(false);
  }
}

}  // namespace test
}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DCT_DCT_TEST_H_
