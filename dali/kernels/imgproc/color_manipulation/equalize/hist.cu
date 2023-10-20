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

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "dali/kernels/imgproc/color_manipulation/equalize/hist.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace hist {

__global__ void ZeroMem(const SampleDesc *sample_descs) {
  auto sample_desc = sample_descs[blockIdx.y];
  sample_desc.out[blockIdx.x * SampleDesc::range_size + threadIdx.x] = 0;
}

__global__ void Histogram(const SampleDesc *sample_descs) {
  // cuda headers do not provide atomicAdd for uint64_t, but they do for unsigned long long int
  using ull_t = unsigned long long int;  // NOLINT(runtime/int)
  static_assert(sizeof(ull_t) == sizeof(uint64_t));
  extern __shared__ char shm[];
  auto *workspace = reinterpret_cast<ull_t *>(shm);
  auto sample_desc = sample_descs[blockIdx.y];
  const uint8_t *in = sample_desc.in;
  auto *out = reinterpret_cast<ull_t *>(sample_desc.out);
  for (unsigned int idx = threadIdx.x; idx < SampleDesc::range_size * sample_desc.num_channels;
       idx += blockDim.x) {
    workspace[idx] = 0;
  }
  __syncthreads();
  for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < sample_desc.num_elements;
       idx += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    uint64_t channel_idx = idx % sample_desc.num_channels;
    atomicAdd(workspace + channel_idx * SampleDesc::range_size + in[idx], 1);
  }
  __syncthreads();
  for (unsigned int idx = threadIdx.x; idx < SampleDesc::range_size * sample_desc.num_channels;
       idx += blockDim.x) {
    atomicAdd(out + idx, workspace[idx]);
  }
}

void HistogramKernelGpu::Run(KernelContext &ctx, TensorListView<StorageGPU, uint64_t, 2> &out,
                             const TensorListView<StorageGPU, const uint8_t, 2> &in) {
  int batch_size = out.num_samples();
  assert(in.num_samples() == batch_size);
  sample_descs_.clear();
  sample_descs_.reserve(batch_size);
  int64_t max_num_blocks = 0;
  int64_t max_num_channels = 0;
  for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
    int64_t num_channels = in.shape[sample_idx][1];
    int64_t num_elements = in.shape[sample_idx].num_elements();
    assert(num_channels == out.shape[sample_idx][0]);
    int64_t num_blocks = div_ceil(num_elements, kBlockSize);
    max_num_blocks = std::max(max_num_blocks, num_blocks);
    max_num_channels = std::max(max_num_channels, num_channels);
    sample_descs_.push_back({out.data[sample_idx], in.data[sample_idx],
                             static_cast<uint64_t>(num_elements),
                             static_cast<uint64_t>(num_channels)});
  }
  int64_t workspace_size = max_num_channels * kShmPerChannelSize;
  if (workspace_size > shared_mem_limit_) {
    throw std::range_error(
        make_string("The maximal number of channels in a sample for histogram is ",
                    shared_mem_limit_ / kShmPerChannelSize, ", however got a sample with ",
                    max_num_channels, "."));
  }
  max_num_blocks = std::min(max_num_blocks, kMaxGridSize);
  SampleDesc *samples_desc_dev;
  std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_);
  dim3 zero_grid{static_cast<unsigned int>(max_num_channels),
                 static_cast<unsigned int>(batch_size)};
  ZeroMem<<<zero_grid, kBlockSize, 0, ctx.gpu.stream>>>(samples_desc_dev);
  CUDA_CALL(cudaGetLastError());
  dim3 hist_grid{static_cast<unsigned int>(max_num_blocks), static_cast<unsigned int>(batch_size)};
  Histogram<<<hist_grid, kBlockSize, workspace_size, ctx.gpu.stream>>>(samples_desc_dev);
  CUDA_CALL(cudaGetLastError());
}

}  // namespace hist
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
