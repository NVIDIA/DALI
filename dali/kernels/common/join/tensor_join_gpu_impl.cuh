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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/core/fast_div.h"
#include "dali/core/math_util.h"

namespace dali {
namespace kernels {
namespace tensor_join {

template <typename Element>
struct InputDesc {
  const Element *data;
  /// Extent of the tensor in the joined axis
  uint64_t join_size;
  /// Combined extent of the joined axes
  uint64_t inner_size;
  /// Stride between slices in the joined axis
  uint64_t outer_stride;

  /**
   * @brief Offset of this tensor's slice in the otput tensor's slice
   *
   * The "slice" here denotes a slice in the concatenation axis.
   */
  uint64_t join_offset;
};

template <typename Element>
struct OutputDesc {
  Element *data;
  /// Stride between slices in the joined axis
  fast_div<uint64_t> outer_stride;
  float guess_tensor_mul;  // used for quickly estimating which tensor we've hit based on the offset

  uint64_t total_size;
};

constexpr int kMaxJoin = (16<<10) / sizeof(InputDesc<void>);

template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float guess_tensor_mul,
                          const InputDesc<Element> *descs, int njoin) {
  int lo = 0, hi = njoin - 1;
  int m = min(static_cast<int>(offset * guess_tensor_mul), njoin - 1);
  while (lo <= hi) {  // binary search with initial guess
    if (offset < descs[m].join_offset)
      hi = m - 1;
    else if (m < njoin-1 && descs[m+1].join_offset >= offset)
      lo = m + 1;
    else
      break;  // Found! For stacking this will always work unless precision fails
    m = (lo + hi) >> 1;  // proceed to regular binary search
  }
  // TODO(michalz): remove
  assert(m >= 0 && m < njoin);
  return m;
}

template <typename Element>
__device__ void JoinTensorsLarge(OutputDesc<Element> out,
                                 const InputDesc<Element> *sh_in,
                                 int njoin) {
  uint64_t out_offset = threadIdx.x + blockDim.x * static_cast<uint64_t>(blockIdx.x);
  uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (; out_offset < out.total_size; out_offset += step) {
    uint64_t join_offset;
    uint64_t outer = div_mod(join_offset, out_offset, out.outer_stride);
    int t = FindTensor(join_offset, out.guess_tensor_mul, sh_in, njoin);
    out.data[out_offset] = sh_in[t].data[join_offset - sh_in[t].join_offset];
  }
}

template <typename Element>
__global__ void JoinTensorsKernel(const OutputDesc<Element> *out,
                                  const InputDesc<Element> *in,
                                  int njoin) {
  int sample_idx = blockIdx.y;
  __shared__ InputDesc<Element> sh_in[kMaxJoin];
  for (int i = threadIdx.x; i < njoin; i += blockDim.x)
    sh_in[i] = in[sample_idx * njoin + i];
  __syncthreads();

  JoinTensorsLarge(out[sample_idx], sh_in, njoin);
}

}  // namespate tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
