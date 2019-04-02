// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <cassert>
#include "dali/kernels/common/scatter_gather.h"

namespace dali {
namespace kernels {

void ScatterGatherGPU::Coalesce() {
  if (ranges_.empty())
    return;
  std::sort(ranges_.begin(), ranges_.end(), [](const CopyRange &a, const CopyRange &b) {
    return a.src < b.src;
  });

  int start = 0;
  int n = ranges_.size();
  bool changed = false;

  // merge
  for (int i = 1; i < n; i++) {
    if (ranges_[i].src == ranges_[start].src + ranges_[start].size &&
        ranges_[i].dst == ranges_[start].dst + ranges_[start].size) {
      ranges_[start].size += ranges_[i].size;
      ranges_[i] = { nullptr, nullptr, 0 };
      changed = true;
    } else {
      start = i;
    }
  }

  if (changed) {
    // compact
    int new_size = 1;  // first item is guaranteed to be non-empty
    for (int i = 1; i < n; i++) {
      if (ranges_[i].size > 0) {
        int j = new_size++;
        if (j != i)
          ranges_[j] = ranges_[i];
      }
    }
    ranges_.resize(new_size);
  }
}

void ScatterGatherGPU::MakeBlocks() {
  size_t max_size = 0;

  for (auto &r : ranges_) {
    if (r.size > max_size)
      max_size = r.size;
  }

  size_per_block_ = std::min(max_size, max_size_per_block_);

  int num_blocks = 0;
  for (auto &r : ranges_)
    num_blocks += (r.size + size_per_block_ - 1) / size_per_block_;

  blocks_.clear();
  blocks_.reserve(num_blocks);
  for (auto &r : ranges_) {
    for (size_t ofs = 0; ofs < r.size; ofs += size_per_block_) {
      blocks_.push_back({ r.src + ofs, r.dst + ofs, std::min(r.size - ofs, size_per_block_) });
    }
  }
  assert(blocks_.size() == num_blocks);

  ReserveGPUBlocks();
}

void ScatterGatherGPU::ReserveGPUBlocks() {
  if (block_capacity_ < blocks_.capacity()) {
    block_capacity_ = blocks_.capacity();
    blocks_dev_ = memory::alloc_unique<CopyRange>(AllocType::GPU, block_capacity_);
  }
}

__global__ void BatchCopy(const ScatterGatherGPU::CopyRange *ranges) {
  auto range = ranges[blockIdx.x];

  for (int i = threadIdx.x; i < range.size; i += blockDim.x) {
    range.dst[i] = range.src[i];
  }
}

void ScatterGatherGPU::Run(cudaStream_t stream) {
  Coalesce();
  if (ranges_.size() <= 2) {
    for (auto &r : ranges_) {
      cudaMemcpyAsync(r.dst, r.src, r.size, cudaMemcpyDeviceToDevice, stream);
    }
    return;
  }

  MakeBlocks();
  cudaMemcpyAsync(blocks_dev_.get(), blocks_.data(), blocks_.size() * sizeof(blocks_[0]),
    cudaMemcpyDeviceToDevice, stream);

  dim3 grid(blocks_.size());
  dim3 block(std::min<size_t>(size_per_block_, 1024));
  BatchCopy<<<grid, block>>>(blocks_dev_.get());
}



}  // namespace kernels
}  // namespace dali
