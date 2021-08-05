// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/data/types.h"

namespace dali {
namespace detail {

__global__ void CopyKernel(uint8_t *dst, const uint8_t *src, int64_t n) {
  int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t i = start; i < n; i += step) {
    dst[i] = src[i];
  }
}

void LaunchCopyKernel(void *dst, const void *src, int64_t nbytes, cudaStream_t stream) {
  unsigned block = std::min<int64_t>(nbytes, 1024);
  unsigned grid = std::min<int64_t>(1024, div_ceil(static_cast<unsigned>(nbytes), block));
  CopyKernel<<<grid, block, 0, stream>>>(reinterpret_cast<uint8_t*>(dst),
                                         reinterpret_cast<const uint8_t*>(src),
                                         nbytes);
  CUDA_CALL(cudaGetLastError());
}

}  // namespace detail
}  // namespace dali
