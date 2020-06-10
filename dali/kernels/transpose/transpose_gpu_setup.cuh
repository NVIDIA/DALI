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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_SETUP_CUH_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_SETUP_CUH_

#include <cuda_runtime.h>
#include <utility>
#include "dali/core/tensor_view.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/transpose/transpose_gpu_impl.cuh"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {
namespace transpose_impl {

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
    TensorShape<> &out_shape, SmallVector<int, static_size> &out_perm,
    const int64_t *shape, const int *perm, int ndim) {
  out_shape = { shape, shape + ndim };
  out_perm = { perm, perm + ndim };

  CollapseUnitDims(out_shape, out_perm);
  CollapseAdjacentDims(out_shape, out_perm);
}

template <typename T>
void InitTiledTranspose(TiledTransposeDesc<T> &desc,
                        const TensorShape<> &shape, span<const int> perm,
                        same_as_t<T> *out = nullptr, same_as_t<const T> *in = nullptr,
                        int grid_size = 1) {
  int ndim = shape.size();

  CalcStrides(desc.in_strides, shape);

  TensorShape<> out_shape = Permute(shape, perm);
  TensorShape<> tmp_strides;
  CalcStrides(tmp_strides, out_shape);
  for (int i = 0; i < ndim; i++) {
    desc.shape[i] = shape[i];
    desc.out_strides[perm[i]] = tmp_strides[i];
  }
  int max_lanes = kTiledTransposeMaxVectorSize / sizeof(T);
  int lanes = 1;
  while (ndim > 0 && perm[ndim-1] == ndim-1) {
    if (lanes * shape[ndim-1] > max_lanes)
      break;
    lanes *= shape[ndim-1];
    ndim--;
  }
  desc.lanes = lanes;
  desc.ndim = ndim;

  // Make sure that when transposing the last dimension, the last two can be transposed in tiles
  // and loaded and stored in contiguous transactions.
  if (desc.out_strides[ndim-2] != lanes) {
    for (int i = 0; i < ndim-2; i++) {
      if (desc.out_strides[i] == lanes) {
        std::rotate(desc.out_strides + i, desc.out_strides + i+1, desc.out_strides + ndim-1);
        std::rotate(desc.in_strides  + i, desc.in_strides  + i+1, desc.in_strides  + ndim-1);
        std::rotate(desc.shape       + i, desc.shape       + i+1, desc.shape       + ndim-1);
        break;
      }
    }
  }

  SmallVector<int64_t, 6> non_tile_dims;
  for (int i = 0; i < ndim - 2; i ++) {
    non_tile_dims.push_back(desc.shape[i]);
  }

  desc.tiles_x = div_ceil(desc.shape[ndim - 1], kTileSize);
  desc.tiles_y = div_ceil(desc.shape[ndim - 2], kTileSize);
  desc.tiles_per_slice = desc.tiles_x * desc.tiles_y;
  desc.total_tiles = volume(non_tile_dims) * desc.tiles_per_slice;
  desc.tiles_per_block = div_ceil(desc.total_tiles, grid_size);

  desc.out = out;
  desc.in = in;
}


template <typename T>
void InitDeinterleave(DeinterleaveDesc<T> &desc,
                      const TensorShape<> &shape, span<const int> perm,
                      same_as_t<T> *out = nullptr, same_as_t<const T> *in = nullptr) {
  int ndim = shape.size();

  CalcStrides(desc.in_strides, shape);

  TensorShape<> out_shape = Permute(shape, perm);
  TensorShape<> tmp_strides;
  CalcStrides(tmp_strides, out_shape);
  for (int i = 0; i < ndim; i++) {
    desc.out_strides[perm[i]] = tmp_strides[i];
  }

  desc.size = volume(shape);
  desc.ndim = ndim;

  desc.out = out;
  desc.in = in;
}

template <typename T>
void InitGenericTranspose(GenericTransposeDesc<T> &desc,
                         const TensorShape<> &shape, span<const int> perm,
                         same_as_t<T> *out = nullptr, same_as_t<const T> *in = nullptr) {
  int ndim = shape.size();

  TensorShape<> out_shape = Permute(shape, perm);
  CalcStrides(desc.out_strides, out_shape);

  TensorShape<> tmp_strides;
  CalcStrides(tmp_strides, shape);
  for (int i = 0; i < ndim; i++) {
    desc.in_strides[i] = tmp_strides[perm[i]];
  }
  if (ndim == 0) {
    desc.out_strides[0] = 0;
    desc.in_strides[0] = 1;
  }

  desc.size = volume(shape);
  desc.ndim = std::max(ndim, 1);

  desc.out = out;
  desc.in = in;
}

}  // namespace transpose_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_SETUP_CUH_
