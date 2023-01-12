// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_EQUALIZED_LUT_CUH_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_EQUALIZED_LUT_CUH_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/sequence_utils.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace lut {
struct SampleDesc {
  static constexpr int range_size = 256;
  static constexpr int range_size_log2 = 8;
  uint8_t *out;
  const int32_t *in;
};

DALI_DEVICE DALI_FORCEINLINE void PrefixSum(const int32_t *__restrict__ in, int32_t *workspace) {
  // load the histogram into shm workspace
  workspace[threadIdx.x] = in[threadIdx.x];
  __syncthreads();

  int stride = 1;
  // prepare partial sums as in-place binary tree
#pragma unroll
  for (int num_active = SampleDesc::range_size >> 1; num_active >= 1; num_active >>= 1) {
    if (threadIdx.x < num_active) {
      int idx = stride * (2 * threadIdx.x + 1) - 1;
      workspace[idx + stride] += workspace[idx];
    }
    __syncthreads();
    stride <<= 1;
  }

#pragma unroll
  for (int pow2 = 2; pow2 <= SampleDesc::range_size >> 1; pow2 <<= 1) {
    stride >>= 1;
    if (threadIdx.x < pow2 - 1) {
      int idx = stride * (threadIdx.x + 1) - 1;
      workspace[idx + (stride >> 1)] += workspace[idx];
    }
    __syncthreads();
  }
}

DALI_DEVICE DALI_FORCEINLINE int32_t FirstNonZero(int32_t *workspace) {
  int32_t begin = 0, end = SampleDesc::range_size - 1;
  while (begin < end) {
    int32_t mid = (begin + end) >> 1;
    if (workspace[mid] != 0) {
      end = mid;
    } else {
      begin = mid + 1;
    }
  }
  return end;
}

__global__ void PrepareLookupTable(const SampleDesc *sample_descs) {
  __shared__ int32_t workspace[SampleDesc::range_size];
  auto sample_desc = sample_descs[blockIdx.x];
  PrefixSum(sample_desc.in, workspace);
  int32_t first_idx = FirstNonZero(workspace);
  int32_t first_val = workspace[first_idx];
  int32_t total = workspace[SampleDesc::range_size - 1];
  float factor = (SampleDesc::range_size - 1.f) / (total - first_val);
  sample_desc.out[threadIdx.x] = ConvertSat<uint8_t>((workspace[threadIdx.x] - first_val) * factor);
}

struct LutKernelGpu {
  static constexpr int kBlockSize = 256;

  void Run(KernelContext &ctx, const TensorListView<StorageGPU, uint8_t, 2> &lut,
           const TensorListView<StorageGPU, const int32_t, 2> &histogram) {
    assert(equalized.num_samples() == histogram.num_samples());
    sample_descs_.clear();
    for (int sample_idx = 0; sample_idx < histogram.num_samples(); sample_idx++) {
      // TODO(ktokarski) use combined range as we have cpp17 now
      auto equalized_channels = sequence_utils::unfolded_view_range<1>(lut[sample_idx]);
      auto hist_channels = sequence_utils::unfolded_view_range<1>(histogram[sample_idx]);
      for (int chunk_idx = 0; chunk_idx < equalized_channels.NumSlices(); chunk_idx++) {
        sample_descs_.push_back(
            {equalized_channels[chunk_idx].data, hist_channels[chunk_idx].data});
      }
    }
    SampleDesc *samples_desc_dev;
    std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_);
    PrepareLookupTable<<<sample_descs_.size(), kBlockSize, 0, ctx.gpu.stream>>>(samples_desc_dev);
  }

  std::vector<SampleDesc> sample_descs_;
};

}  // namespace lut
}  // namespace equalize
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_EQUALIZED_LUT_CUH_
