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
  fast_div<uint64_t> outer_stride;
  fast_div<uint64_t> join_stride;
  float guess_tensor_mul;  // used for quickly estimating which tensor we've hit based on the offset

  uint64_t total_size;
  uint64_t outer_size;
};

constexpr int kMaxShmJoin = (32<<10) / sizeof(InputDesc<void>);

template <typename Element>
DALI_HOST_DEV int FindTensor(uint64_t offset, float guess_tensor_mul,
                          const InputDesc<Element> *__restrict__ descs, int njoin) {
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
  return m;
}

template <typename Element>
__device__ void JoinTensors(OutputDesc<Element> out,
                            const InputDesc<Element> *__restrict__ in,
                            int njoin) {
  uint64_t block_size = static_cast<uint64_t>(blockDim.x) * blockDim.y;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  uint64_t out_offset = tid + block_size * blockIdx.x;
  uint64_t step = block_size * gridDim.x;
  for (; out_offset < out.total_size; out_offset += step) {
    uint64_t join_offset;
    uint64_t outer = div_mod(join_offset, out_offset, out.outer_stride);
    int t = FindTensor(join_offset, out.guess_tensor_mul, in, njoin);
    ptrdiff_t offset_in_tensor = join_offset - in[t].join_offset;
    out.data[out_offset] = in[t].data[offset_in_tensor + outer * in[t].outer_stride];
  }
}

template <typename Element>
__global__ void JoinTensorsKernel(const OutputDesc<Element> *__restrict__ out,
                                  const InputDesc<Element> *__restrict__ in,
                                  int njoin) {
  int sample_idx = blockIdx.y;
  JoinTensors(out[sample_idx], in + sample_idx * njoin, njoin);
}


template <typename Element>
__device__ void StackTensors(OutputDesc<Element> out,
                             const InputDesc<Element> *__restrict__ in,
                             int njoin) {
  uint64_t in_size = in[0].outer_stride * out.outer_size;

  uint64_t uniform_offset = static_cast<uint64_t>(blockDim.x) * blockIdx.x;
  uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  __shared__ Element tmp[32][33];
  __shared__ uint64_t inner[32], outer[32];

  for (; uniform_offset < in_size; uniform_offset += step) {
    auto in_offset = uniform_offset + threadIdx.x;

    outer[threadIdx.x] = div_mod(inner[threadIdx.x], in_offset, out.join_stride);

    for (int t0 = 0; t0 < njoin; t0 += 32) {
      int t1 = min(t0 + 32, njoin);

      __syncthreads();

      if (in_offset < in_size) {
        for (int t = t0 + threadIdx.y; t < t1; t += blockDim.y) {
          tmp[t - t0][threadIdx.x] = in[t].data[in_offset];
        }
      }
      __syncthreads();


      int t = t0 + threadIdx.x;
      if (t < njoin) {
        for (int y = threadIdx.y; y < 32 ; y += blockDim.y) {
          auto tr_offset = uniform_offset + y;
          if (tr_offset >= in_size)
            break;

          out.data[out.outer_stride * outer[y] + t * out.join_stride + inner[y]] =
            tmp[threadIdx.x][y];
        }
      }
    }
  }
}

template <typename Element>
__global__ void StackTensorsKernel(const OutputDesc<Element> *__restrict__ out,
                                   const InputDesc<Element> *__restrict__ in,
                                   int njoin) {
  int sample_idx = blockIdx.y;
  StackTensors(out[sample_idx], in + njoin * sample_idx, njoin);
}


}  // namespate tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
