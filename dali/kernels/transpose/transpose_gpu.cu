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

#include "dali/kernels/transpose/transpose_gpu_impl.cuh"
#include "dali/kernels/transpose/transpose_gpu_setup.cuh"

namespace dali {
namespace kernels {
namespace transpose_impl {

enum class TransposeMethod {
  Copy = 0,
  Generic,
  Tiled,
  Deinterleave
};

struct TransposeInfo {
  int                 element_size;
  TransposeMethod     method;
  TensorShape<>       shape;
  SmallVector<int, 6> perm;
};

constexpr int kMaxDeinterleaveSize = 32;

static bool UseTiledTranspose(
    const int64_t *shape, const int *perm, int ndim, int element_size ) {
  int xdim = ndim-1;
  int ydim = 0;
  for (; ydim < xdim; ydim++) {
    if (perm[ydim] == xdim)
      break;
  }
  float tile_coverage = shape[xdim] * shape[ydim];
  tile_coverage /= align_up(shape[xdim], kTileSize) * align_up(shape[ydim], kTileSize);
  return tile_coverage > 0.7f;
}

static TransposeMethod GetTransposeMethod(
    const int64_t *shape, const int *perm, int ndim, int element_size) {
  if (ndim == 1)
    return TransposeMethod::Copy;
  if (perm[ndim-1] == ndim - 1) {
    assert(ndim >= 3);
    if (shape[ndim-1] * element_size < kTiledTransposeMaxVectorSize) {
      if (UseTiledTranspose(shape, perm, ndim-1, element_size))
        return TransposeMethod::Tiled;
    }
  } else {
    if (UseTiledTranspose(shape, perm, ndim, element_size))
      return TransposeMethod::Tiled;
    else if (shape[ndim-1] * element_size <= kMaxDeinterleaveSize)
      return TransposeMethod::Deinterleave;
  }
  return TransposeMethod::Generic;
}

void GetTransposeInfo(TransposeInfo &info, int element_size,
                      span<const int64_t> in_shape, span<const int> perm) {
  SimplifyPermute(info.shape, info.perm, in_shape.data(), perm.data(), in_shape.size());
  info.element_size = element_size;
  int ndim = info.shape.size();
  info.method = GetTransposeMethod(info.shape.data(), info.perm.data(), ndim, element_size);
}

void GetTransposeInfo(TransposeInfo *infos, int element_size,
                      const TensorListShape<> &tls, span<const int> perm) {
  int N = tls.num_samples();
  for (int i = 0; i < N; i++) {
    GetTransposeInfo(infos[i], element_size, tls.tensor_shape_span(i), perm);
  }
}



}  // namespace transpose_impl
}  // namespace kernels
}  // namespace dali
