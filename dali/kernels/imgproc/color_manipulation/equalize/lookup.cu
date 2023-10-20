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

#include "dali/kernels/imgproc/color_manipulation/equalize/lookup.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace lookup {

__global__ void Lookup(const SampleDesc *sample_descs) {
  auto sample_desc = sample_descs[blockIdx.y];
  for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < sample_desc.num_elements;
       idx += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint8_t *in = sample_desc.in;
    uint8_t *out = sample_desc.out;
    uint64_t channel_idx = idx % sample_desc.num_channels;
    out[idx] = __ldg(sample_desc.lut + channel_idx * SampleDesc::range_size + in[idx]);
  }
}

void LookupKernelGpu::Run(KernelContext &ctx, const TensorListView<StorageGPU, uint8_t, 2> &out,
                          const TensorListView<StorageGPU, const uint8_t, 2> &in,
                          const TensorListView<StorageGPU, const uint8_t, 2> &lut) {
  int batch_size = out.num_samples();
  assert(in.num_samples() == batch_size && lut.num_samples() == batch_size);
  sample_descs_.clear();
  sample_descs_.reserve(batch_size);
  int64_t max_num_blocks = 0;
  for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
    int64_t num_channels = in.shape[sample_idx][1];
    int64_t num_elements = in.shape[sample_idx].num_elements();
    assert(num_channels == lut.shape[sample_idx][0] && num_channels == out.shape[sample_idx][1]);
    int64_t num_blocks = div_ceil(num_elements, kBlockSize);
    max_num_blocks = std::max(max_num_blocks, num_blocks);
    sample_descs_.push_back({out.data[sample_idx], in.data[sample_idx], lut.data[sample_idx],
                             static_cast<uint64_t>(num_elements),
                             static_cast<uint64_t>(num_channels)});
  }
  max_num_blocks = std::min(max_num_blocks, kMaxGridSize);
  SampleDesc *samples_desc_dev;
  std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_);
  dim3 grid{static_cast<unsigned int>(max_num_blocks), static_cast<unsigned int>(batch_size)};
  Lookup<<<grid, kBlockSize, 0, ctx.gpu.stream>>>(samples_desc_dev);
  CUDA_CALL(cudaGetLastError());
}

}  // namespace lookup
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
