// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "dali/core/cuda_error.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {
namespace kernels {

namespace detail {
size_t Coalesce(span<CopyRange> ranges) {
  if (ranges.empty())
    return 0;
  std::sort(ranges.begin(), ranges.end(), [](const CopyRange &a, const CopyRange &b) {
    return a.src < b.src;
  });

  int start = 0;
  size_t n = ranges.size();
  bool changed = false;

  // merge
  for (size_t i = 1; i < n; i++) {
    if (ranges[i].src == ranges[start].src + ranges[start].size &&
        ranges[i].dst == ranges[start].dst + ranges[start].size) {
      ranges[start].size += ranges[i].size;
      ranges[i] = { nullptr, nullptr, 0 };
      changed = true;
    } else {
      start = i;
    }
  }

  if (changed) {
    // compact
    size_t new_size = 1;  // first item is guaranteed to be non-empty
    for (size_t i = 1; i < n; i++) {
      if (ranges[i].size > 0) {
        size_t j = new_size++;
        if (j != i)
          ranges[j] = ranges[i];
      }
    }
    return new_size;
  }
  return n;
}

}  // namespace detail

size_t ScatterGatherBase::MakeBlocks(std::vector<CopyRange> &blocks,
                                     const std::vector<CopyRange> &ranges) {
  size_t max_size = 0;

  for (auto &r : ranges) {
    if (r.size > max_size)
      max_size = r.size;
  }

  size_t size_per_block = std::min(max_size, max_size_per_block_);

  size_t num_blocks = 0;
  for (auto &r : ranges)
    num_blocks += (r.size + size_per_block - 1) / size_per_block;

  blocks.clear();
  blocks.reserve(num_blocks);
  for (auto &r : ranges) {
    for (size_t ofs = 0; ofs < r.size; ofs += size_per_block) {
      blocks.push_back({ r.src + ofs, r.dst + ofs, std::min(r.size - ofs, size_per_block) });
    }
  }
  assert(blocks.size() == num_blocks);
  return size_per_block;
}

void ScatterGatherGPU::MakeBlocks() {
  size_per_block_ = ScatterGatherBase::MakeBlocks(blocks_, ranges_);
}

void ScatterGatherCPU::MakeBlocks(size_t blocks_lower_limit) {
  size_t heap_size = std::max(blocks_lower_limit, ranges_.size());

  heap_.clear();
  heap_.reserve(heap_size);
  heap_ = ranges_;

  auto cmp = [](CopyRange &left, CopyRange &right) { return left.size < right.size; };

  std::make_heap(heap_.begin(), heap_.end(), cmp);

  while (heap_.size() < blocks_lower_limit) {
    // Take out the larges
    std::pop_heap(heap_.begin(), heap_.end(), cmp);
    CopyRange largest = heap_.back();
    heap_.pop_back();

    // Split the range into halfs
    CopyRange first_half = {largest.src, largest.dst, largest.size / 2};
    CopyRange second_half = {largest.src + first_half.size, largest.dst + first_half.size,
                             largest.size - first_half.size};

    // Add back to the heap
    heap_.push_back(first_half);
    std::push_heap(heap_.begin(), heap_.end(), cmp);
    heap_.push_back(second_half);
    std::push_heap(heap_.begin(), heap_.end(), cmp);
  }

  ScatterGatherBase::MakeBlocks(blocks_, heap_);
}


__global__ void BatchCopy(const ScatterGatherBase::CopyRange *ranges) {
  auto range = ranges[blockIdx.x];

  for (size_t i = threadIdx.x; i < range.size; i += blockDim.x) {
    range.dst[i] = range.src[i];
  }
}

void ScatterGatherGPU::Run(cudaStream_t stream, bool reset, ScatterGatherGPU::Method method,
                           cudaMemcpyKind memcpyKind) {
  Coalesce();

  // TODO(michalz): Error handling

  bool use_memcpy = (method == ScatterGatherGPU::Method::Memcpy) ||
    (method == ScatterGatherGPU::Method::Default && ranges_.size() <= 2);

  if (use_memcpy) {
    for (auto &r : ranges_) {
      CUDA_CALL(cudaMemcpyAsync(r.dst, r.src, r.size, memcpyKind, stream));
    }
  } else {
    MakeBlocks();
    blocks_dev_.from_host(blocks_, stream);

    dim3 grid(blocks_.size());
    dim3 block(std::min<size_t>(size_per_block_, 1024));
    BatchCopy<<<grid, block, 0, stream>>>(blocks_dev_.data());
    CUDA_CALL(cudaGetLastError());
  }

  if (reset)
    Reset();
}

}  // namespace kernels
}  // namespace dali
