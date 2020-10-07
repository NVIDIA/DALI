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
  /**
   * @brief Offset of this tensor's slice in the otput tensor's slice
   *
   * The "slice" here denotes a slice in the concatenation axis.
   */
  uint64_t join_offset;

  /// Distance between joined slices
  uint64_t outer_stride;

  const Element *__restrict__ data;
};

template <typename Element>
struct OutputDesc {
  Element *__restrict__ data;
  fast_div<uint64_t> outer_stride;
  float guess_tensor_mul;  // used for quickly estimating which tensor we've hit based on the offset

  uint64_t total_size;
};

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
  uint64_t block_size = static_cast<uint64_t>(blockDim.x);
  uint64_t out_offset = threadIdx.x + block_size * blockIdx.x;
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


template <typename ElementType, int out_ndim, int in_ndim>
void FillDescs(span<OutputDesc<ElementType>> output_descs,
               span<InputDesc<ElementType>> input_descs,
               const OutListGPU<ElementType, out_ndim> &output,
               span<const InListGPU<ElementType, in_ndim> *const> inputs,
               int axis) {
  int njoin = inputs.size();
  int N = output.num_samples();
  assert(static_cast<int>(output_descs.size()) == N);
  assert(static_cast<int>(input_descs.size()) == N * njoin);

  for (int i = 0; i < N; i++) {
    int64_t join_offset = 0;
    for (int t = 0; t < njoin; t++) {
      auto in_shape = inputs[t]->tensor_shape_span(i);
      auto &desc = input_descs[i*njoin+t];
      desc.data = inputs[t]->data[i];
      desc.outer_stride = volume(in_shape.begin() + axis, in_shape.end());
      desc.join_offset = join_offset;
      join_offset += desc.outer_stride;
    }

    auto out_shape = output.tensor_shape_span(i);
    auto &out_desc = output_descs[i];
    out_desc.data = output.data[i];
    auto join_size = volume(out_shape.begin() + axis, out_shape.end());
    out_desc.outer_stride = join_size;
    out_desc.total_size = volume(out_shape);
    if (njoin == 2) {
      // Special case - if there are just two tensors, we can set the multiplier
      // so that it reaches 1 when the offset is within the second tensor. The fact
      // that it reach values >= njoin is not a problem, since the value is clamped anyway to
      // safeguard against loss of precision.
      out_desc.guess_tensor_mul = input_descs[1].join_offset ? 1.0 / input_descs[1].join_offset : 0;
    } else {
      out_desc.guess_tensor_mul = 1.0 * njoin / join_size;
    }
  }
}


}  // namespate tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_CUH_
