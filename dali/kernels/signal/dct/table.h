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

#ifndef DALI_KERNELS_SIGNAL_DCT_TABLE_H_
#define DALI_KERNELS_SIGNAL_DCT_TABLE_H_

#include <cmath>
#include "dali/kernels/signal/dct/dct_args.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

template <typename T>
void FillCosineTableTypeI(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  assert(input_length > 1);
  assert(!normalize);
  double phase_mul = M_PI / (input_length - 1);
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    table[idx++] = 0.5;  // n = 0
    for (int64_t n = 1; n < input_length-1; n++) {
      table[idx++] = std::cos(phase_mul * k * n);
    }
    table[idx++] = k % 2 == 0 ?  0.5 : -0.5;  // n = input_length - 1
  }
}

template <typename T>
void FillCosineTableTypeII(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  double phase_mul = M_PI / input_length;
  double factor_k_0 = 1, factor_k_i = 1;
  if (normalize) {
    factor_k_i = std::sqrt(2.0 / input_length);
    factor_k_0 = 1.0 / std::sqrt(input_length);
  }
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    double norm_factor = (k == 0) ? factor_k_0 : factor_k_i;
    for (int64_t n = 0; n < input_length; n++) {
      table[idx++] = norm_factor * std::cos(phase_mul * (n + 0.5) * k);
    }
  }
}


template <typename T>
void FillCosineTableTypeIII(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  double phase_mul = M_PI / input_length;
  double factor_n_0 = 0.5, factor_n_i = 1;
  if (normalize) {
    factor_n_i = std::sqrt(2.0 / input_length);
    factor_n_0 = 1.0 / std::sqrt(input_length);
  }
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    table[idx++] = factor_n_0;  // n = 0
    for (int64_t n = 1; n < input_length; n++) {
      table[idx++] = factor_n_i * std::cos(phase_mul * n * (k + 0.5));
    }
  }
}


template <typename T>
void FillCosineTableTypeIV(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  double phase_mul = M_PI / input_length;
  double factor = normalize ? std::sqrt(2.0 / input_length) : 1.0;
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    for (int64_t n = 0; n < input_length; n++) {
      table[idx++] = factor * std::cos(phase_mul * (n + 0.5) * (k + 0.5));
    }
  }
}


template <typename T>
void FillCosineTable(T *table, int64_t input_length, DctArgs args) {
  switch (args.dct_type) {
    case 1:
      FillCosineTableTypeI(table, input_length, args.ndct, args.normalize);
      break;
    case 2:
      FillCosineTableTypeII(table, input_length, args.ndct, args.normalize);
      break;
    case 3:
      FillCosineTableTypeIII(table, input_length, args.ndct, args.normalize);
      break;
    case 4:
      FillCosineTableTypeIV(table, input_length, args.ndct, args.normalize);
      break;
    default:
      assert(false);
  }
}

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DCT_TABLE_H_
