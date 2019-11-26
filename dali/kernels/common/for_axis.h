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

#ifndef DALI_KERNELS_COMMON_FOR_AXIS_H_
#define DALI_KERNELS_COMMON_FOR_AXIS_H_

#include <utility>

namespace dali {
namespace kernels {

/**
 * @brief iterator through all the 1-dimensional slices on a given axis
 */
template <typename OutputType, typename InputType, typename Functor>
void ForAxis(OutputType *out_ptr,
             const InputType *in_ptr,
             const int64_t *out_shape,
             const int64_t *out_strides,
             const int64_t *in_shape,
             const int64_t *in_strides,
             int axis,
             int ndim,
             Functor &&func,
             int current_dim = 0) {
  if (current_dim == ndim) {
    func(out_ptr, in_ptr, out_shape[axis], out_strides[axis], in_shape[axis], in_strides[axis]);
    return;
  }

  if (axis == current_dim) {
    ForAxis(out_ptr, in_ptr, out_shape, out_strides, in_shape, in_strides,
            axis, ndim, std::forward<Functor>(func), current_dim+1);
  } else {
    for (int i = 0; i < in_shape[current_dim]; i++) {
      ForAxis(out_ptr + i * out_strides[current_dim],
              in_ptr + i * in_strides[current_dim],
              out_shape, out_strides,
              in_shape, in_strides,
              axis, ndim, std::forward<Functor>(func), current_dim+1);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_FOR_AXIS_H_
