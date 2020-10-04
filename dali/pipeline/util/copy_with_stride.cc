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

#include "dali/pipeline/util/copy_with_stride.h"
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
                                 const Index *in_strides,
                                 const Index *out_strides,
                                 const Index *shape,
                                 Index ndim,
                                 Index dim, Index deepest_contiguous) {
  auto out_ptr = reinterpret_cast<uint8*>(output);
  auto in_ptr = reinterpret_cast<const uint8*>(input);
  const auto item_size = out_strides[ndim - 1];
  if (dim == deepest_contiguous) {
    std::memcpy(out_ptr, in_ptr,
        volume(shape + dim, shape + ndim)*item_size);
    return;
  }
  if (dim == ndim - 1) {
    VALUE_SWITCH(item_size, elem_size, (1, 2, 3, 4, 5, 6, 7, 8, 12, 16),
                 (CopyVecStatic<elem_size>(out_ptr, in_ptr, shape[ndim - 1], in_strides[ndim - 1])),
                 (CopyVec(out_ptr, in_ptr, shape[ndim - 1], in_strides[ndim - 1], item_size)));
    return;
  }
  const auto out_stride = out_strides[dim];
  const auto in_stride = in_strides[dim];
  const auto n = shape[dim];
  for (Index i = 0; i < n; ++i) {
    CopyWithStrideHelper(out_ptr, in_ptr, in_strides, out_strides,
        shape, ndim, dim + 1, deepest_contiguous);
    out_ptr += out_stride;
    in_ptr += in_stride;
  }
}

inline Index DeepestContiguous(const Index *in_strides,
                               const Index *shape,
                               int ndim,
                               size_t item_size) {
  ssize_t dim_prod = 1;
  for (int i = ndim-1; i >= 0; --i) {
    if (in_strides[i] != dim_prod*static_cast<Index>(item_size)) {
      return i+1;
    }
    dim_prod *= shape[i];
  }
  return 0;
}

template <>
void CopyWithStride<CPUBackend>(void *output, const void *input,
                    const Index *in_strides,
                    const Index *shape,
                    int ndim,
                    size_t item_size,
                    cudaStream_t) {
  assert(ndim >= 0);
  if (!in_strides) {
    std::memcpy(output, input, item_size * volume(shape, shape + ndim));
    return;
  }
  std::vector<Index> out_strides(ndim);
  out_strides.back() = item_size;
  for (int i = ndim - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * shape[i + 1];
  }
  auto deepest_contiguous = DeepestContiguous(in_strides, shape, ndim, item_size);
  CopyWithStrideHelper(output, input, in_strides, out_strides.data(),
                       shape, ndim, 0, deepest_contiguous);
}

}  // namespace dali
