// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_utils.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {

namespace cast {

struct SampleDesc {
  void *output;
  const void *input;
  uint32_t first_block;
  int64_t sample_size;
};

inline __device__ uint32_t FindSampleIdx(const SampleDesc *samples,
                                         unsigned nsamples) {
  uint32_t i = 0;
  for (uint32_t jump = (1 << (32 - __clz(nsamples) - 1)); jump; jump >>= 1) {
    if (i + jump < nsamples && samples[i + jump].first_block <= blockIdx.x)
      i += jump;
  }
  return i;
}

template <typename Out, typename In>
__global__ void BinSearchCastKernel(const SampleDesc *samples,
                                    unsigned nsamples, int block_sz) {
  int sample_idx = FindSampleIdx(samples, nsamples);
  SampleDesc sample = samples[sample_idx];
  auto *out = static_cast<Out *>(sample.output);
  const auto *in = static_cast<const In *>(sample.input);
  auto size = sample.sample_size;
  auto block_idx = blockIdx.x - sample.first_block;
  auto block_start = block_idx * block_sz;
  auto block_end = cuda_min<int64_t>(block_start + block_sz, size);
  for (unsigned x = threadIdx.x + block_start; x < block_end; x += blockDim.x) {
    out[x] = ConvertSat<Out>(in[x]);
  }
}

}  // namespace cast
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_COMMON_CAST_CUH_
