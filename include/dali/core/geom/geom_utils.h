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

#ifndef DALI_CORE_GEOM_GEOM_UTILS_H_
#define DALI_CORE_GEOM_GEOM_UTILS_H_

#include "dali/core/geom/vec.h"
#include "dali/core/geom/mat.h"
#include "dali/core/tensor_view.h"

namespace dali {

template <int N, typename T = float>
vec<N, T> as_vec(TensorView<StorageCPU, const T, 1> view) {
  if (view.num_elements() == 1) {
    return vec<N, T>(view.data[0]);
  }
  assert(N == view.num_elements());
  return *reinterpret_cast<const vec<N, T>*>(view.data);
}

template <int N, typename T = float>
vec<N, T> as_vec(TensorView<StorageCPU, const T, DynamicDimensions> view) {
  return as_vec<N, T>(view.template to_static<1>());
}

template <int N, int M, typename T = float>
mat<N, M, T> as_mat(TensorView<StorageCPU, const T, 2> view) {
  assert(N * M == view.num_elements());
  return *reinterpret_cast<const mat<N, M, T>*>(view.data);
}

template <int N, int M, typename T = float>
mat<N, M, T> as_mat(TensorView<StorageCPU, const T, DynamicDimensions> view) {
  return as_mat<N, M, T>(view.template to_static<2>());
}

}  // namespace dali

#endif  // DALI_CORE_GEOM_GEOM_UTILS_H_
