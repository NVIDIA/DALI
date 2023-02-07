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

#include <vector>

#include "dali/kernels/imgproc/color_manipulation/equalize/lut.h"
#include "dali/pipeline/data/sequence_utils.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace lut {

DALI_DEVICE DALI_FORCEINLINE void PrefixSum(uint64_t *workspace, const uint64_t *__restrict__ in) {
  // load the histogram into shm workspace
  workspace[threadIdx.x] = in[threadIdx.x];
  __syncthreads();

  int stride = 1;
  // prepare partial sums as in-place binary tree
#pragma unroll
  for (unsigned int num_active = SampleDesc::range_size >> 1; num_active >= 1; num_active >>= 1) {
    if (threadIdx.x < num_active) {
      int idx = stride * (2 * threadIdx.x + 1) - 1;
      workspace[idx + stride] += workspace[idx];
    }
    __syncthreads();
    stride <<= 1;
  }

#pragma unroll
  for (unsigned int pow2 = 2; pow2 <= SampleDesc::range_size >> 1; pow2 <<= 1) {
    stride >>= 1;
    if (threadIdx.x < pow2 - 1) {
      int idx = stride * (threadIdx.x + 1) - 1;
      workspace[idx + (stride >> 1)] += workspace[idx];
    }
    __syncthreads();
  }
}

DALI_DEVICE DALI_FORCEINLINE int32_t FirstNonZero(uint64_t *workspace) {
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
  __shared__ uint64_t workspace[SampleDesc::range_size];
  auto sample_desc = sample_descs[blockIdx.x];
  PrefixSum(workspace, sample_desc.in);
  int32_t first_idx = FirstNonZero(workspace);
  uint64_t first_val = workspace[first_idx];
  uint64_t total = workspace[SampleDesc::range_size - 1];
  if (first_val == total) {
    sample_desc.out[threadIdx.x] = threadIdx.x;
  } else {
    double factor = (SampleDesc::range_size - 1.) / (total - first_val);
    sample_desc.out[threadIdx.x] =
        ConvertSat<uint8_t>((workspace[threadIdx.x] - first_val) * factor);
  }
}

void LutKernelGpu::Run(KernelContext &ctx, const TensorListView<StorageGPU, uint8_t, 2> &lut,
                       const TensorListView<StorageGPU, const uint64_t, 2> &histogram) {
  assert(lut.num_samples() == histogram.num_samples());
  sample_descs_.clear();
  for (int sample_idx = 0; sample_idx < histogram.num_samples(); sample_idx++) {
    auto ranges = sequence_utils::unfolded_views_range<1>(lut[sample_idx], histogram[sample_idx]);
    for (auto &&[lu, bin] : ranges) {
      sample_descs_.push_back({lu.data, bin.data});
    }
  }
  SampleDesc *samples_desc_dev;
  std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_);
  PrepareLookupTable<<<sample_descs_.size(), kBlockSize, 0, ctx.gpu.stream>>>(samples_desc_dev);
  CUDA_CALL(cudaGetLastError());
}

}  // namespace lut
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
