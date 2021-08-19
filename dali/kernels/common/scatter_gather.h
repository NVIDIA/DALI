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

#ifndef DALI_KERNELS_COMMON_SCATTER_GATHER_H_
#define DALI_KERNELS_COMMON_SCATTER_GATHER_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include "dali/core/api_helper.h"
#include "dali/core/span.h"
#include "dali/kernels/alloc.h"

namespace dali {
namespace kernels {

namespace detail {
struct DLL_PUBLIC CopyRange {
  const char *src;
  char *dst;
  size_t size;
};

DLL_PUBLIC size_t Coalesce(span<CopyRange> ranges);
}  // namespace detail

/**
 * Base class for ScatterGather with the CopyRange handling
 */
class DLL_PUBLIC ScatterGatherBase {
 public:
  static constexpr size_t kDefaultBlockSize = 64<<10;

  ScatterGatherBase() = default;

  ScatterGatherBase(size_t max_size_per_block, size_t estimated_num_blocks)
  : max_size_per_block_(max_size_per_block) {
    blocks_.reserve(estimated_num_blocks);
  }

  explicit ScatterGatherBase(size_t max_size_per_block)
      : ScatterGatherBase(max_size_per_block, 0) {}

  ScatterGatherBase(size_t max_size_per_block, size_t total_size, size_t num_ranges)
  : ScatterGatherBase(
      max_size_per_block,
      (total_size + num_ranges * (max_size_per_block - 1)) / max_size_per_block) {
    ranges_.reserve(num_ranges);
  }

  /**
   * @brief Clear any registered range copies
   */
  void Reset() {
    ranges_.clear();
    blocks_.clear();
  }

  /**
   * @brief Adds one copy to the batch
   */
  void AddCopy(void *dst, const void *src, size_t size) {
    if (size > 0) {
      ranges_.push_back({static_cast<const char *>(src), static_cast<char *>(dst), size});
    }
  }

  enum class Method {
    Default = 0,  // For GPU, uses scatter-gather kernel, unless there are 2 or fewer single
                  // effective copy ranges, in that cases cudaMemcpyAsync is used
                  // For CPU, uses memcpy
    Memcpy = 1,   // Always use cudaMemcpyAsync, only for GPU
    Kernel = 2,   // Always use scatter-gather kernel, only for GPU
  };

  using CopyRange = detail::CopyRange;

 protected:
  std::vector<CopyRange> ranges_;

  /**
   * @brief Sorts and merges contiguous ranges
   */
  void Coalesce() {
    size_t n = detail::Coalesce(make_span(ranges_.data(), ranges_.size()));
    ranges_.resize(n);
  }

  /**
   * @brief Divides ranges so they don't exceed `max_block_size_`
   */
  void MakeBlocks();

  size_t max_size_per_block_ = kDefaultBlockSize;
  std::vector<CopyRange> blocks_;
  size_t block_capacity_ = 0;
  size_t size_per_block_ = 0;
};


/**
 * Implements a device-to-device batch copy of multiple sources to multiple destinations
 */
class DLL_PUBLIC ScatterGatherGPU : public ScatterGatherBase {
 public:
  static constexpr size_t kDefaultBlockSize = 64<<10;

  ScatterGatherGPU() = default;

  ScatterGatherGPU(size_t max_size_per_block, size_t estimated_num_blocks)
      : ScatterGatherBase(max_size_per_block, estimated_num_blocks) {
    ReserveGPUBlocks();
  }

  explicit ScatterGatherGPU(size_t max_size_per_block) : ScatterGatherBase(max_size_per_block) {
    ReserveGPUBlocks();
  }

  ScatterGatherGPU(size_t max_size_per_block, size_t total_size, size_t num_ranges)
      : ScatterGatherBase(max_size_per_block, total_size, num_ranges) {
    ReserveGPUBlocks();
  }

  /**
   * @brief Executes the copies
   * @param stream     - the cudaStream on which the copies are scheduled
   * @param reset      - if true, calls Reset after processing is over
   * @param method     - see ScatterGatherGPU::CopyMethod
   * @param memcpyKind - determines the cudaMemcpyKind when using cudaMemcpy
   */
  DLL_PUBLIC void
  Run(cudaStream_t stream, bool reset = true, Method method = Method::Default,
      cudaMemcpyKind memcpyKind = cudaMemcpyDefault);

  using CopyRange = detail::CopyRange;

 private:
  /**
   * @brief Reserves GPU memory for the description of the blocks.
   */
  void ReserveGPUBlocks();

  kernels::memory::KernelUniquePtr<CopyRange> blocks_dev_;
};


/**
 * Implements a batch copy of multiple sources to multiple destinations for CPU using thread pool
 */
class DLL_PUBLIC ScatterGatherCPU : public ScatterGatherBase {
 public:
  static constexpr size_t kDefaultBlockSize = 64<<10;

  ScatterGatherCPU() = default;

  ScatterGatherCPU(size_t max_size_per_block, size_t estimated_num_blocks)
      : ScatterGatherBase(max_size_per_block, estimated_num_blocks) {}

  explicit ScatterGatherCPU(size_t max_size_per_block) : ScatterGatherBase(max_size_per_block) {}

  ScatterGatherCPU(size_t max_size_per_block, size_t total_size, size_t num_ranges)
      : ScatterGatherBase(max_size_per_block, total_size, num_ranges) {}


  /**
   * @brief Executes the copies
   * @param exec_engine - pool to run the copies in
   * @param reset       - if true, calls Reset after processing is over
   */
  template <typename ExecutionEngine>
  DLL_PUBLIC void Run(ExecutionEngine &exec_engine, bool reset = true) {
    Coalesce();
    for (auto &r : ranges_) {
      exec_engine.AddWork([=](int thread_id) { std::memcpy(r.dst, r.src, r.size); }, r.size);
    }
    exec_engine.RunAll();

    if (reset)
      Reset();
  }

  using CopyRange = detail::CopyRange;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SCATTER_GATHER_H_
