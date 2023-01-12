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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_HIST_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_HIST_H_

#include "dali/core/common.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

#include "dali/npp/npp.h"


namespace dali {
namespace kernels {
namespace equalize_hist {

struct SampleDesc {
  static constexpr int range_size = 256;
  int32_t *out;
  const uint8_t *in;
  int width;
  fast_div<uint32_t> num_channels;
};

__global__ void ZeroMem(const SampleDesc *sample_descs) {
  auto sample_desc = sample_descs[blockIdx.y];
  sample_desc.out[blockIdx.x * SampleDesc::range_size + threadIdx.x] = 0;
}

__global__ void Histogram(const SampleDesc *sample_descs) {
  extern __shared__ char shm[];
  auto *workspace = reinterpret_cast<int32_t *>(shm);
  auto sample_desc = sample_descs[blockIdx.y];
  const uint8_t *in = sample_desc.in;
  int32_t *out = sample_desc.out;
  for (int idx = threadIdx.x; idx < SampleDesc::range_size * sample_desc.num_channels;
       idx += blockDim.x) {
    workspace[idx] = 0;
  }
  __syncthreads();
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sample_desc.width;
       idx += blockDim.x * gridDim.x) {
    uint32_t channel_idx;
    div_mod(channel_idx, idx, sample_desc.num_channels);
    atomicAdd(workspace + channel_idx * SampleDesc::range_size + in[idx], 1);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < SampleDesc::range_size * sample_desc.num_channels;
       idx += blockDim.x) {
    atomicAdd(out + idx, workspace[idx]);
  }
}

}  // namespace equalize_hist

struct HistogramKernelGpu {
  static constexpr unsigned int kBlockSize = 256;
  static constexpr unsigned int kMaxGridSize = 128;

  void Run(KernelContext &ctx, TensorListView<StorageGPU, int32_t, 2> &out,
           const TensorListView<StorageGPU, const uint8_t, 2> &in) {
    int batch_size = out.num_samples();
    assert(in.num_samples() == batch_size);
    sample_descs_.clear();
    sample_descs_.reserve(batch_size);
    unsigned int max_num_blocks = 0;
    unsigned int max_num_channels = 0;
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      unsigned int num_channels = in.shape[sample_idx][1];
      int width = in.shape[sample_idx][0] * num_channels;
      assert(num_channels == out.shape[sample_idx][1]);
      unsigned int num_blocks = div_ceil(width, kBlockSize);
      max_num_blocks = std::max(max_num_blocks, num_blocks);
      max_num_channels = std::max(max_num_channels, num_channels);
      sample_descs_.push_back({out.data[sample_idx], in.data[sample_idx], width, num_channels});
    }
    max_num_blocks = std::min(max_num_blocks, kMaxGridSize);
    equalize_hist::SampleDesc *samples_desc_dev;
    std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_);
    dim3 zero_grid{max_num_channels, static_cast<unsigned int>(batch_size)};
    equalize_hist::ZeroMem<<<zero_grid, kBlockSize, 0, ctx.gpu.stream>>>(samples_desc_dev);
    dim3 hist_grid{max_num_blocks, static_cast<unsigned int>(batch_size)};
    int workspace_size = max_num_channels * equalize_hist::SampleDesc::range_size * sizeof(int32_t);
    equalize_hist::Histogram<<<hist_grid, kBlockSize, workspace_size, ctx.gpu.stream>>>(
        samples_desc_dev);
  }

 protected:
  std::vector<equalize_hist::SampleDesc> sample_descs_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_HIST_H_
