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

#ifndef DALI_KERNELS_COMMON_SCATTER_GATHER_H_
#define DALI_KERNELS_COMMON_SCATTER_GATHER_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "dali/kernels/alloc.h"
#include "dali/kernels/span.h"
#include "dali/api_helper.h"

namespace dali {
namespace kernels {

namespace detail {
struct CopyRange {
  const char *src;
  char *dst;
  size_t size;
};

size_t Coalesce(span<CopyRange> ranges);
}  // namespace detail

/// Implements a device-to-device batch copy of multiple sources to multiple destinations
class DLL_PUBLIC ScatterGatherGPU {
 public:
  static constexpr size_t kDefaultBlockSize = 64<<10;

  ScatterGatherGPU() = default;

  ScatterGatherGPU(size_t max_size_per_block, size_t estimated_num_blocks)
  : max_size_per_block_(max_size_per_block) {
    blocks_.reserve(estimated_num_blocks);
    ReserveGPUBlocks();
  }

  explicit ScatterGatherGPU(size_t max_size_per_block) : ScatterGatherGPU(max_size_per_block, 0) {}

  ScatterGatherGPU(size_t max_size_per_block, size_t total_size, size_t num_ranges)
  : ScatterGatherGPU(
      max_size_per_block,
      (total_size + num_ranges * (max_size_per_block - 1)) / max_size_per_block) {
    ranges_.reserve(num_ranges);
  }

  void Reset() {
    ranges_.clear();
    blocks_.clear();
  }

  /// @brief Adds one copy to the batch
  void AddCopy(void *dst, const void *src, size_t size) {
    if (size > 0) {
      ranges_.push_back({
        static_cast<const char*>(src),
        static_cast<char*>(dst),
        size
      });
    }
  }

  /// @brief Executes the copies
  /// @param stream - the cudaStream on which the copies are scheduled
  /// @param reset - if true, calls Reset after processing is over
  /// @param useMemcpyOnly - if true, all copies are executed using cudaMemcpy;
  ///                        otherwise a batched kernel is used if there are more than 2 ranges
  DLL_PUBLIC void Run(cudaStream_t stream, bool reset = true, bool useMemcpyOnly = false);

  using CopyRange = detail::CopyRange;

 private:
  std::vector<CopyRange> ranges_;

  /// @brief Sorts and merges contiguous ranges
  void Coalesce() {
    size_t n = detail::Coalesce(make_span(ranges_.data(), ranges_.size()));
    ranges_.resize(n);
  }

  /// @brief Divides ranges so they don't exceed `max_block_size_`
  void MakeBlocks();

  /// @brief Reserves GPU memory for the description of the blocks.
  void ReserveGPUBlocks();

  size_t max_size_per_block_ = kDefaultBlockSize;
  std::vector<CopyRange> blocks_;
  kernels::memory::KernelUniquePtr<CopyRange> blocks_dev_;
  size_t block_capacity_ = 0;
  size_t size_per_block_ = 0;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SCATTER_GATHER_H_
