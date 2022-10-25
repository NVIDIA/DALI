// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <utility>
#include "dali/core/api_helper.h"
#include "dali/core/fast_div.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/transpose/transpose_gpu_impl.cuh"
#include "dali/kernels/transpose/transpose_util.h"

namespace dali {
namespace kernels {
namespace transpose_impl {

enum class TransposeMethod {
  Copy = 0,
  Generic,
  Tiled,
  Interleave,
  Deinterleave
};

DLL_PUBLIC TransposeMethod GetTransposeMethod(const int64_t *shape,
                                              const int *perm,
                                              int ndim,
                                              int element_size);

template <typename T>
void InitTiledTranspose(TiledTransposeDesc<T> &desc,
                        const TensorShape<> &shape, span<const int> perm,
                        same_as_t<T> *out = nullptr, same_as_t<const T> *in = nullptr,
                        int grid_x_size = 0) {
  int ndim = shape.size();

  CalcStrides(desc.in_strides, shape);

  TensorShape<> out_shape = permute(shape, perm);
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
  if (desc.out_strides[ndim-2] != static_cast<uint64_t>(lanes)) {
    for (int i = 0; i < ndim-2; i++) {
      if (desc.out_strides[i] == static_cast<uint64_t>(lanes)) {
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
  // make sure that it's when grid_x_size is 0, the result is garbage
  desc.tiles_per_block = grid_x_size ? div_ceil(desc.total_tiles, grid_x_size) : 0;

  desc.out = out;
  desc.in = in;
}

template <typename T>
void UpdateTiledTranspose(TiledTransposeDesc<T> &desc,
                          same_as_t<T> *out, same_as_t<const T> *in, int grid_x_size) {
  assert(out != nullptr);
  assert(in  != nullptr);
  assert(grid_x_size > 0);
  desc.out = out;
  desc.in = in;
  desc.tiles_per_block = grid_x_size ? div_ceil(desc.total_tiles, grid_x_size) : 0;
}


template <typename T>
void InitDeinterleave(DeinterleaveDesc<T> &desc,
                      const TensorShape<> &shape, span<const int> perm,
                      same_as_t<T> *out = nullptr, same_as_t<const T> *in = nullptr) {
  int ndim = shape.size();

  CalcStrides(desc.in_strides, shape);

  TensorShape<> out_shape = permute(shape, perm);
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

  TensorShape<> out_shape = permute(shape, perm);
  CalcStrides(desc.out_strides, out_shape);

  TensorShape<> tmp_strides;
  CalcStrides(tmp_strides, shape);
  assert(static_cast<size_t>(perm.size()) < dali::size(desc.in_strides));
  permute(make_span(desc.in_strides, perm.size()), tmp_strides, perm);

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
