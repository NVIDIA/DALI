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

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "dali/kernels/alloc.h"
#include "dali/api_helper.h"

namespace dali {
namespace kernels {

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
      (total_size + num_ranges * (max_size_per_block - 1)) / max_size_per_block)
  {}

  void Reset() {
    ranges_.clear();
    blocks_.clear();
  }

  void AddCopy(void *dst, const void *src, size_t size) {
    ranges_.push_back({
      static_cast<const char*>(src),
      static_cast<char*>(dst),
      size
    });
  }
  DLL_PUBLIC void Run(cudaStream_t stream);

  struct CopyRange {
    const char *src;
    char *dst;
    size_t size;
  };
 private:
  std::vector<CopyRange> ranges_;

  void Coalesce();
  void MakeBlocks();
  void ReserveGPUBlocks();

  size_t max_size_per_block_ = kDefaultBlockSize;
  std::vector<CopyRange> blocks_;
  kernels::memory::KernelUniquePtr<CopyRange> blocks_dev_;
  size_t block_capacity_ = 0;
  size_t size_per_block_ = 0;

};

}  // namespace kernels
}  // namespace dali
