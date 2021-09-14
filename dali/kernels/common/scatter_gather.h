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
#include <limits>
#include <queue>
#include <vector>
#include "dali/core/api_helper.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/span.h"

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
  // Special value to indicate that blocks should not be splitted
  static constexpr size_t kAnyBlockSize = std::numeric_limits<size_t>::max();

  ScatterGatherBase() = default;

  explicit ScatterGatherBase(size_t max_size_per_block) : max_size_per_block_(max_size_per_block) {}

  ScatterGatherBase(size_t max_size_per_block, size_t total_size, size_t num_ranges)
      : max_size_per_block_(max_size_per_block) {
    ranges_.reserve(num_ranges);
  }

  /**
   * @brief Clear any registered range copies
   */
  void Reset() {
    ranges_.clear();
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
  size_t MakeBlocks(std::vector<CopyRange> &blocks, const std::vector<CopyRange> &ranges);

  size_t max_size_per_block_ = kDefaultBlockSize;
};


/**
 * Implements a device-to-device batch copy of multiple sources to multiple destinations
 */
class DLL_PUBLIC ScatterGatherGPU : public ScatterGatherBase {
 public:
  static constexpr size_t kDefaultBlockSize = 64<<10;

  ScatterGatherGPU() = default;

  ScatterGatherGPU(size_t max_size_per_block, size_t estimated_num_blocks)
      : ScatterGatherBase(max_size_per_block) {
    blocks_.reserve(estimated_num_blocks);
    blocks_dev_.reserve(estimated_num_blocks);
  }

  explicit ScatterGatherGPU(size_t max_size_per_block) : ScatterGatherBase(max_size_per_block) {}

  ScatterGatherGPU(size_t max_size_per_block, size_t total_size, size_t num_ranges)
      : ScatterGatherGPU(max_size_per_block, (total_size + num_ranges * (max_size_per_block - 1)) /
                                                 max_size_per_block) {}

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

  /**
   * @brief Clear any registered range copies
   */
  void Reset() {
    ScatterGatherBase::Reset();
    blocks_.clear();
    blocks_dev_.clear();
  }

  using CopyRange = detail::CopyRange;

 private:
  /**
   * @brief Divides ranges so they don't exceed `max_block_size_`
   */
  void MakeBlocks();

  std::vector<CopyRange> blocks_;
  DeviceBuffer<CopyRange> blocks_dev_;
  size_t size_per_block_ = 0;
};


/**
 * Implements a batch copy of multiple sources to multiple destinations for CPU using thread pool
 */
class DLL_PUBLIC ScatterGatherCPU : public ScatterGatherBase {
 public:
  ScatterGatherCPU() = default;

  ScatterGatherCPU(size_t max_size_per_block, size_t estimated_num_blocks)
      : ScatterGatherBase(max_size_per_block) {
    heap_.resize(estimated_num_blocks);
    blocks_.resize(estimated_num_blocks);
  }

  explicit ScatterGatherCPU(size_t max_size_per_block) : ScatterGatherBase(max_size_per_block) {}

  ScatterGatherCPU(size_t max_size_per_block, size_t total_size, size_t num_ranges)
      : ScatterGatherCPU(max_size_per_block, (total_size + num_ranges * (max_size_per_block - 1)) /
                                                 max_size_per_block) {}

  /**
   * @brief Executes the copies
   * @param exec_engine - pool to run the copies in
   * @param reset       - if true, calls Reset after processing is over
   */
  template <typename ExecutionEngine>
  void Run(ExecutionEngine &exec_engine, bool reset = true) {
    Coalesce();

    size_t total_size = 0;
    for (auto &r : ranges_) {
      total_size += r.size;
    }

    if (total_size < kSmallSizeThreshold) {
      for (auto &r : ranges_) {
        std::memcpy(r.dst, r.src, r.size);
      }
    } else {
      MakeBlocks(exec_engine.NumThreads() * kTasksMultiplier);
      for (auto &r : blocks_) {
        exec_engine.AddWork([=](int thread_id) { std::memcpy(r.dst, r.src, r.size); }, r.size);
      }
      exec_engine.RunAll();
    }

    if (reset)
      Reset();
  }

  void Reset() {
    ScatterGatherBase::Reset();
    heap_.clear();
    blocks_.clear();
  }

  using CopyRange = detail::CopyRange;

 private:
  /**
   * @brief Divides ranges so there are at least `blocks_lower_limit` elements that don't exceed
   * `max_block_size_`.
   */
  void MakeBlocks(size_t blocks_lower_limit);

  std::vector<CopyRange> heap_;
  std::vector<CopyRange> blocks_;

  // Sizes below this threshold will be copied without using the pool/execution engine
  static constexpr size_t kSmallSizeThreshold = 1 << 11;
  // At least how many more tasks we want compared to the number of worker threads in exec engine
  static constexpr size_t kTasksMultiplier = 3;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SCATTER_GATHER_H_
