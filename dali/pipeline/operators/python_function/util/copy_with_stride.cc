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

#include "dali/pipeline/operators/python_function/util/copy_with_stride.h"
#include <cstring>
#include <cassert>
#include "dali/core/static_switch.h"
#include "dali/core/util.h"

namespace dali {

inline void CopyVec(uint8 *output, const uint8 *input, size_t elems,
                    size_t in_stride, size_t item_size) {
  for (size_t i = 0; i < elems; ++i) {
    for (size_t j = 0; j < item_size; j++)
      *output++ = input[j];
    input += in_stride;
  }
}

template <size_t item_size>
inline void CopyVecStatic(uint8 *output, const uint8 *input, size_t elems, size_t in_stride) {
  for (size_t i = 0; i < elems; ++i) {
    for (size_t j = 0; j < item_size; j++)
      output[j] = input[j];
    output += item_size;
    input += in_stride;
  }
}

inline void CopyWithStrideHelper(void *output, const void *input,
                                 const std::vector<Index> &in_strides,
                                 const std::vector<Index> &out_strides,
                                 const std::vector<Index> &shape,
                                 Index dim, Index deepest_contiguous) {
  auto out_ptr = reinterpret_cast<uint8*>(output);
  auto in_ptr = reinterpret_cast<const uint8*>(input);
  if (dim == deepest_contiguous) {
    std::memcpy(out_ptr, in_ptr,
        volume(shape.begin() + dim, shape.end())*out_strides.back());
    return;
  }
  if (static_cast<size_t>(dim) == shape.size() - 1) {
    VALUE_SWITCH(out_strides.back(), elem_size, (1, 2, 3, 4, 5, 6, 7, 8, 12, 16),
                 (CopyVecStatic<elem_size>(out_ptr, in_ptr, shape.back(), in_strides.back())),
                 (CopyVec(out_ptr, in_ptr, shape.back(), in_strides.back(), out_strides.back())));
    return;
  }
  const auto out_stride = out_strides[dim];
  const auto in_stride = in_strides[dim];
  const auto n = shape[dim];
  for (Index i = 0; i < n; ++i) {
    CopyWithStrideHelper(out_ptr, in_ptr, in_strides, out_strides,
        shape, dim + 1, deepest_contiguous);
    out_ptr += out_stride;
    in_ptr += in_stride;
  }
}

inline Index DeepestContiguous(const std::vector<Index>& in_strides,
                        const std::vector<Index>& shape,
                        size_t item_size) {
  ssize_t dim_prod = 1;
  for (int i = in_strides.size()-1; i >= 0; --i) {
    if (in_strides[i] != dim_prod*static_cast<Index>(item_size)) {
      return i+1;
    }
    dim_prod *= shape[i];
  }
  return 0;
}

void CopyWithStride(void *output, const void *input,
                    const std::vector<Index>& in_strides,
                    const std::vector<Index>& shape,
                    size_t item_size) {
  assert(!shape.empty());
  assert(in_strides.size() == shape.size());
  std::vector<Index> out_strides(shape.size());
  out_strides.back() = item_size;
  for (Index i = shape.size() - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * shape[i + 1];
  }
  auto deepest_contiguous = DeepestContiguous(in_strides, shape, item_size);
  CopyWithStrideHelper(output, input, in_strides, out_strides, shape, 0, deepest_contiguous);
}

}  // namespace dali
