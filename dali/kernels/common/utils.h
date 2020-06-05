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

#ifndef DALI_KERNELS_COMMON_UTILS_H_
#define DALI_KERNELS_COMMON_UTILS_H_

#include <utility>
#include "dali/core/host_dev.h"

namespace dali {
namespace kernels {

template <typename T, typename U>
T Permute(const T& in, const U& permutation) {
  T out = in;
  for (size_t i = 0; i < permutation.size(); i++) {
    auto idx = permutation[i];
    out[i] = in[idx];
  }
  return out;
}

template <typename Shape>
DALI_HOST_DEV
void CalcStrides(Shape &strides, const Shape& shape) {
  strides[strides.size() - 1] = 1;
  for (int d = strides.size() - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * shape[d + 1];
  }
}

template <typename OutStrides, typename Shape>
DALI_HOST_DEV
void CalcStrides(OutStrides &out_strides, const Shape& shape) {
  Shape strides = shape;  // strides has the same size as shape
  CalcStrides(strides, shape);
  for (int d = 0; d < static_cast<int>(strides.size()); d++)
    out_strides[d] = strides[d];
}

template <typename Shape, typename OutShape = Shape>
DALI_HOST_DEV
Shape GetStrides(const Shape& shape) {
  Shape strides = shape;
  CalcStrides(strides, shape);
  return strides;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_UTILS_H_
