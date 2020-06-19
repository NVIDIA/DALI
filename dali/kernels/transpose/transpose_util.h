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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_UTIL_H_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_UTIL_H_

#include <cstdint>
#include <utility>
#include "dali/core/small_vector.h"
#include "dali/core/tensor_shape.h"

namespace dali {
namespace kernels {
namespace transpose_impl {

template <int ndim1, int ndim2>
void Permute(TensorListShape<ndim1> &out, const TensorListShape<ndim2> &in,
             span<const int> permutation) {
  dali::detail::check_compatible_ndim<ndim1, ndim2>();
  int N = in.num_samples();
  int ndim = in.sample_dim();
  out.resize(N, ndim);
  assert(permutation.size() == ndim);
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < ndim; d++) {
      out.tensor_shape_span(i)[d] = in.tensor_shape_span(i)[permutation[d]];
    }
  }
}

template <size_t static_size>
inline void CollapseUnitDims(TensorShape<> &shape, SmallVector<int, static_size> &perm) {
  int ndim = shape.size();
  SmallVector<int, static_size> dim_map, out_perm;
  TensorShape<> out_shape;
  dim_map.resize(ndim, -1);
  for (int i = 0; i < ndim; i++) {
    if (shape[i] != 1) {
      dim_map[i] = out_shape.size();
      out_shape.shape.push_back(shape[i]);
    }
  }
  for (int i = 0; i < ndim; i++) {
    int m = dim_map[perm[i]];
    if (m >= 0)
      out_perm.push_back(m);
  }
  shape = std::move(out_shape);
  perm = std::move(out_perm);
}

template <size_t static_size>
inline void CollapseAdjacentDims(TensorShape<> &shape, SmallVector<int, static_size> &perm) {
  int ndim = shape.size();

  SmallVector<int, static_size> dim_map, inv_perm, out_perm;
  TensorShape<> out_shape;

  dim_map.resize(ndim, -1);

  inv_perm.resize(ndim);
  for (int i = 0; i < ndim; i++)
    inv_perm[perm[i]] = i;

  int nkeep = 0;
  int64_t extent = 1;
  int last = 0;
  for (int i = 0; i < ndim; i++) {
    if (inv_perm[i] != inv_perm[last] + i - last) {
      dim_map[i] = nkeep++;
      out_shape.shape.push_back(extent);
      last = i;
      extent = shape[i];
    } else {
      extent *= shape[i];
    }
    if (nkeep == 0) {
      dim_map[i] = nkeep++;
    }
  }

  out_shape.shape.push_back(extent);

  for (int i = 0; i < ndim; i++) {
    int pmap = dim_map[perm[i]];
    if (pmap >= 0)
      out_perm.push_back(pmap);
  }

  shape = std::move(out_shape);
  perm = std::move(out_perm);
}

template <size_t static_size>
inline void SimplifyPermute(
    TensorShape<> &simplified_shape, SmallVector<int, static_size> &simplified_perm,
    const int64_t *shape, const int *perm, int ndim) {
  simplified_shape = { shape, shape + ndim };
  simplified_perm = { perm, perm + ndim };

  CollapseUnitDims(simplified_shape, simplified_perm);
  CollapseAdjacentDims(simplified_shape, simplified_perm);
}

template <size_t static_size, int in_ndim>
inline void SimplifyPermute(
    TensorShape<> &simplified_shape, SmallVector<int, static_size> &simplified_perm,
    const TensorShape<in_ndim> &shape, span<const int> perm) {
  assert(static_cast<int>(perm.size()) == shape.size());
  SimplifyPermute(simplified_shape, simplified_perm, shape.data(), perm.data(), shape.size());
}

}  // namespace transpose_impl
}  // namespace kernels
}  // namespace dali

#endif  //  DALI_KERNELS_TRANSPOSE_TRANSPOSE_UTIL_H_
