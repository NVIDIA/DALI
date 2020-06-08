#// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_TRANSPOSE_GPU_IMPL_CUH_
#define DALI_KERNELS_COMMON_TRANSPOSE_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/core/tensor_view.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/common/transpose_gpu_def.h"

namespace dali {
namespace kernels {
namespace transpose_impl {


struct TransposeInfo {
  void                *out;
  const void          *in;
  TransposeMethod     method;
  TensorShape<>       shape;
  SmallVector<int, 6> perm;
};

template <typename T>
struct TiledTransposeDesc {
  T *__restrict__ out;
  const T *__restrict__ in;
  uint64_t in_strides[kMaxNDim];
  // the output strides are permuted and it's this permutation that defines the whole operation
  uint64_t out_strides[kMaxNDim];
  fast_div<uint64_t> shape[kMaxNDim];

  uint32_t tiles_y;
  uint32_t tiles_per_block;
  uint32_t total_tiles;
  int ndim;

  fast_div<uint32_t> tiles_x;
  fast_div<uint32_t> tiles_per_slice;
};

template <typename T>
__device__ void TransposeTiled(const TiledTransposeDesc<T> &desc) {
  int ndim = desc.ndim;

  unsigned start_tile = blockIdx.x * desc.tiles_per_block;
  unsigned end_tile = min(desc.total_tiles, start_tile + desc.tiles_per_block);
  unsigned tile_in_slice;
  uint64_t fused_slice = desc.ndim > 2
    ? div_mod(tile_in_slice, start_tile, desc.tiles_per_slice)
    : 0;

  uint64_t pos[kMaxNDim];

  for (int d = ndim - 3; d >= 0; d--) {
    fused_slice = div_mod(pos[d], fused_slice, desc.shape[d]);
  }

  unsigned tile_x, tile_y;
  tile_y = div_mod(tile_x, tile_in_slice, desc.tiles_x);
  pos[ndim - 1] = tile_x * kTileSize;
  pos[ndim - 2] = tile_y * kTileSize;

  for (uint64_t tile = start_tile; tile < end_tile; tile++) {
    uint64_t in_ofs = 0, out_ofs = 0;
    for (int d = 0; d < ndim - 2; d++) {
      in_ofs  += desc.in_strides[d] * pos[d];
      out_ofs += desc.out_strides[d] * pos[d];

    }
    int64_t in_x  = pos[ndim-1] + threadIdx.x;
    int64_t in_y  = pos[ndim-2] + threadIdx.y;
    int64_t out_x = pos[ndim-1] + threadIdx.y;
    int64_t out_y = pos[ndim-2] + threadIdx.x;

    // These two lines work regardless of the permutation (memory transactions may be inefficient,
    // but the result is correct)
    in_ofs  += desc.in_strides[ndim-2]  * in_y  + desc.in_strides[ndim-1]  * in_x;
    out_ofs += desc.out_strides[ndim-2] * out_y + desc.out_strides[ndim-1] * out_x;
    // These two assume that the tile can be read as contiguous rows and written as contiguous
    // columns - i.e. in_strides[ndim-1] == 1 and out_strides[ndim-2] == 1
    // in_ofs  += desc.in_strides[ndim-2]  * in_y  + in_x;
    // out_ofs += out_y + desc.out_strides[ndim-1] * out_x;

    __syncthreads();
    __shared__ T tmp[kTileSize][kTileSize + 1];
    int tile_w = min(static_cast<uint64_t>(kTileSize), desc.shape[ndim-1] - pos[ndim-1]);
    int tile_h = min(static_cast<uint64_t>(kTileSize), desc.shape[ndim-2] - pos[ndim-2]);
    if (threadIdx.x < tile_w) {
      for (int ty = threadIdx.y, dy = 0; ty < tile_h; ty += blockDim.y, dy += blockDim.y) {
        tmp[ty][threadIdx.x] = desc.in[in_ofs + desc.in_strides[ndim-2]*dy];
      }
    }
    __syncthreads();

    if (threadIdx.x < tile_h) {
      for (int ty = threadIdx.y, dy = 0; ty < tile_w; ty += blockDim.y, dy += blockDim.y) {
        desc.out[out_ofs + desc.out_strides[ndim-1]*dy] = tmp[threadIdx.x][ty];
      }
    }

    for (int d = ndim - 1; d >= 0; d--) {
      uint64_t delta = d < ndim-2 ? 1 : kTileSize;  // inner two dimensions are tiled
      pos[d] += delta;
      if (pos[d] < desc.shape[d])
        break;
      pos[d] = 0;
    }
  }
}

template <typename T>
__global__ void TransposeTiledSingle(TiledTransposeDesc<T> desc) {
  TransposeTiled(desc);
}

template <typename T>
__global__ void TransposeTiledBatch(const TiledTransposeDesc<T> *descs) {
  TransposeTiled(descs[blockIdx.y]);
}

}  // namespace transpose_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_TRANSPOSE_GPU_IMPL_CUH_
