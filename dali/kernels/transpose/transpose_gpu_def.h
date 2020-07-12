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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_DEF_H_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_DEF_H_

namespace dali {
namespace kernels {
namespace transpose_impl {

static constexpr int kMaxNDim = 16;
static constexpr int kTileSize = 32;
static constexpr int kTiledTransposeMaxVectorSize = 32;

static constexpr int kTiledTransposeMaxSharedMem =
  kTiledTransposeMaxVectorSize * kTileSize * (kTileSize+1);

static_assert(kTiledTransposeMaxSharedMem <= 48<<10,
  "Tile won't fit in shared memory on some supported archs.");

}  // namespace transpose_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_DEF_H_
