// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_CAST_CUH_
#define DALI_KERNELS_COMMON_CAST_CUH_

#include "dali/core/convert.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {

struct CastSampleDesc {
  void *output;
  const void *input;
};

struct CastSampleBlockDesc {
  int first_block;
  int sample_size;
};

template <typename OType, typename IType>
__device__ __forceinline__ void CastKernelInternal(const CastSampleDesc& sample,
                                                   int block_start, int block_end) {
  auto *out = static_cast<OType *>(sample.output);
  const auto *in = static_cast<const IType *>(sample.input);
  for (int x = threadIdx.x + block_start; x < block_end; x += blockDim.x) {
    out[x] = ConvertSat<OType>(in[x]);
  }
}

template <typename OType, typename IType>
__global__ void BatchedCastKernel(const CastSampleDesc *samples, const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  CastKernelInternal<OType, IType>(sample, block.start.x, block.end.x);
}

template <typename OType, typename IType>
__global__ void BinSearchCastKernel(const CastSampleDesc *samples,
                                    const CastSampleBlockDesc *params,
                                    int nsamples, int block_volume_scale) {
  int i = 0;
  for (int jump = (1 << (32 - __clz(nsamples) - 1)); jump; jump /= 2) {
    if (i + jump < nsamples && params[i + jump].first_block <= blockIdx.x)
      i += jump;
  }
  CastSampleDesc sample = samples[i];
  int size = params[i].sample_size;
  int block_offset = blockIdx.x - params[i].first_block;

  int block_size = block_volume_scale * blockDim.x;
  int block_start = block_offset * block_size;
  int block_end = block_start + block_size <= size ? block_start + block_size : size;

  CastKernelInternal<OType, IType>(sample, block_start, block_end);
}

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_COMMON_CAST_CUH_
