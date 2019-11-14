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

#ifndef DALI_KERNELS_SIGNAL_SIGNAL_KERNEL_UTILS_H_
#define DALI_KERNELS_SIGNAL_SIGNAL_KERNEL_UTILS_H_

#include <utility>
#include <vector>

namespace dali {
namespace kernels {
namespace signal {

inline int64_t next_pow2(int64_t n) {
  int64_t pow2 = 2;
  while (n > pow2) {
    pow2 *= 2;
  }
  return pow2;
}

inline bool is_pow2(int64_t n) {
  return (n & (n-1)) == 0;
}

template <typename OutputType, typename InputType>
void Get1DSlices(std::vector<std::pair<OutputType*, const InputType*>>& slices,
                 const int64_t *out_shape,
                 const int64_t *out_strides,
                 const int64_t *in_shape,
                 const int64_t *in_strides,
                 int axis,
                 int ndim) {
  for (int dim = 0; dim < ndim; dim++) {
    if (axis != dim) {
      int sz = slices.size();
      for (int i = 0; i < sz; i++) {
        auto &slice = slices[i];
        auto *out_ptr = slice.first;
        auto *in_ptr = slice.second;
        for (int i = 1; i < in_shape[dim]; i++) {
          out_ptr += out_strides[dim];
          in_ptr += in_strides[dim];
          slices.push_back({out_ptr, in_ptr});
        }
      }
    }
  }
}

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

template <typename Shape>
Shape GetStrides(const Shape& shape) {
  Shape strides = shape;
  strides[strides.size()-1] = 1;
  for (int d = strides.size()-2; d >= 0; d--) {
    strides[d] = strides[d+1] * shape[d+1];
  }
  return strides;
}

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_SIGNAL_KERNEL_UTILS_H_
