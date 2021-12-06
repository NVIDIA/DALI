// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/util.h"
#include "dali/core/traits.h"

namespace dali {
namespace kernels {

template <typename Stride, typename Extent>
inline void CalcStrides(Stride *strides, const Extent *shape, int ndim) {
  if (ndim > 0) {
    uint64_t v = 1;
    for (int i = ndim - 1; i > 0; i--) {
      strides[i] = v;
      v *= shape[i];
    }
    strides[0] = v;
  }
}


template <typename Strides, typename Shape>
DALI_HOST_DEV
void CalcStrides(Strides &strides, const Shape& shape) {
  int ndim = dali::size(shape);
  resize_if_possible(strides, ndim);  // no-op if strides is a plain array or std::array
  if (ndim > 0) {
    int64_t v = 1;
    for (int d = ndim - 1; d > 0; d--) {
      strides[d] = v;
      v *= shape[d];
    }
    strides[0] = v;
  }
}

template <typename Shape, typename OutShape = Shape>
DALI_HOST_DEV
OutShape GetStrides(const Shape& shape) {
  OutShape strides = shape;
  CalcStrides(strides, shape);
  return strides;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_UTILS_H_
