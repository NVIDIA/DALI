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
  const Element *__restrict__ data;
  /// Extent of the tensor in the joined axis
  uint64_t join_size;
  /// Distance between joined slices
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
  Element *__restrict__ data;
  /// Stride between slices in the joined axis
  fast_div<uint64_t> outer_stride;
  float guess_tensor_mul;  // used for quickly estimating which tensor we've hit based on the offset

  uint64_t total_size;
  uint64_t outer_size;
};

constexpr int kMaxShmJoin = (32<<10) / sizeof(InputDesc<void>);

template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float guess_tensor_mul,
                          const InputDesc<Element> *descs, int njoin) {
  int lo = 0, hi = njoin - 1;
  int m = min(floor_int((offset + 0.5f) * guess_tensor_mul), njoin - 1);
  do {  // binary search with initial guess
    if (offset < descs[m].join_offset)
      hi = m - 1;
    else if (offset >= descs[m].join_offset + descs[m].outer_stride)
      lo = m + 1;
    else
      break;  // Found! For stacking this will always work unless precision fails
    m = (lo + hi) >> 1;  // proceed to regular binary search
  } while (lo <= hi);
  // TODO(michalz): remove
  assert(m >= 0 && m < njoin);
  return m;
}

template <typename Element>
__device__ void JoinTensors(OutputDesc<Element> out,
                            const InputDesc<Element> *__restrict__ sh_in,
                            int njoin) {
  uint64_t block_size = static_cast<uint64_t>(blockDim.x) * blockDim.y;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  uint64_t out_offset = tid + block_size * blockIdx.x;
  uint64_t step = block_size * gridDim.x;
  for (; out_offset < out.total_size; out_offset += step) {
    uint64_t join_offset;
    uint64_t outer = div_mod(join_offset, out_offset, out.outer_stride);
    int t = FindTensor(join_offset, out.guess_tensor_mul, sh_in, njoin);
    ptrdiff_t offset_in_tensor = join_offset - sh_in[t].join_offset;
    assert(offset_in_tensor >= 0);  // TODO(michalz): remove
    out.data[out_offset] = sh_in[t].data[offset_in_tensor + outer * sh_in[t].outer_stride];
  }
}

template <typename Element>
__global__ void JoinTensorsKernel(const OutputDesc<Element> *out,
                                  const InputDesc<Element> *in,
                                  int njoin) {
  int sample_idx = blockIdx.y;
  JoinTensors(out[sample_idx], in + sample_idx * njoin, njoin);
}


template <typename Element>
__device__ void StackTensors(OutputDesc<Element> out,
                             const InputDesc<Element> *in,
                             fast_div<uint32_t> njoin) {
  int warp_idx = threadIdx.x >> 5;
  int lane_idx = threadIdx.x & 31;
  int warps_per_block = blockDim.x >> 5;

  uint32_t y = threadIdx.y + blockIdx.x * blockDim.y;
  int t = y % njoin;

  __shared__ Element tmp[32][33];

  uint64_t in_offset = 32 * static_cast<uint64_t>(blockIdx.x);
  uint64_t step = static_cast<uint64_t>(32) * gridDim.x;
  for (; in_offset < out.outer_stride; in_offset += step) {

    __syncthreads();
  }
}

template <typename Element>
__global__ void StackTensorsKernel(const OutputDesc<Element> *out,
                                   const InputDesc<Element> *in,
                                   fast_div<uint32_t> njoin) {
  int sample_idx = blockIdx.y;
  StackTensors(out[sample_idx], in, njoin);
}


}  // namespate tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
