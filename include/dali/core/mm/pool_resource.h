// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_MM_POOL_RESOURCE_H_
#define DALI_CORE_MM_POOL_RESOURCE_H_

#include <mutex>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace mm {

struct pool_options {
  /**
   * @brief Maximum block size
   *
   * Growth stops at this point; larger blocks are allocated only when allocate is called with
   * a larger memory requirements.
   */
  size_t max_block_size = static_cast<size_t>(-1);  // no limit
  /// Minimum size of blocks requested from upstream
  size_t min_block_size = (1 << 12);
  /// The factor by which the allocation size grows until it reaches max_block_size
  float growth_factor = 2;
  /**
   * @brief Whether to try to allocate smaller blocks from upstream if default upcoming
   *        block is unavailable.
   */
  bool try_smaller_on_failure = true;
  size_t upstream_alignment = 256;
};

constexpr pool_options default_host_pool_opts() noexcept {
  return { (1 << 28), (1 << 12), 2.0f, true };
}

constexpr pool_options default_device_pool_opts() noexcept {
  return { (static_cast<size_t>(1) << 32), (1 << 20), 2.0f, false };
}

template <class FreeList, class LockType>
class pool_resource_base : public memory_resource {
 public:
  explicit pool_resource_base(memory_resource *upstream = nullptr, const pool_options opt = {})
  : upstream_(upstream), options_(opt) {
     next_block_size_ = opt.min_block_size;
  }

  pool_resource_base(const pool_resource_base &) = delete;
  pool_resource_base(pool_resource_base &&) = delete;

  ~pool_resource_base() {
    free_all();
  }

  void free_all() {
    for (auto &block : blocks_) {
      upstream_->deallocate(block.ptr, block.bytes, block.alignment);
    }
    blocks_.clear();
    free_list_.clear();
  }

 protected:
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (!bytes)
      return nullptr;

    {
      lock_guard guard(lock_);
      void *ptr = free_list_.get(bytes, alignment);
      if (ptr)
        return ptr;
    }
    alignment = std::max(alignment, options_.upstream_alignment);
    size_t blk_size = bytes;
    void *new_block = get_upstream_block(blk_size, bytes, alignment);
    assert(new_block);
    try {
      lock_guard guard(lock_);
      blocks_.push_back({ new_block, blk_size, alignment });
      if (blk_size == bytes) {
        // we've allocated a block exactly of the required size - there's little
        // chance that it will be merged with anything in the pool, so we'll return it as-is
        return new_block;
      } else {
        // we've allocated an oversized block - put the remainder in the free list
        lock_guard guard(lock_);
        free_list_.put(static_cast<char *>(new_block) + bytes, blk_size - bytes);
        return new_block;
      }
    } catch (...) {
      upstream_->deallocate(new_block, blk_size, alignment);
      throw;
    }
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    lock_guard guard(lock_);
    free_list_.put(ptr, bytes);
  }

  void *get_upstream_block(size_t &blk_size, size_t min_bytes, size_t alignment) {
    blk_size = next_block_size(min_bytes);
    for (;;) {
      try {
        return upstream_->allocate(blk_size, alignment);
      } catch (const std::bad_alloc &) {
        if (blk_size == min_bytes || !options_.try_smaller_on_failure)
          throw;
        blk_size = std::max(min_bytes, blk_size >> 1);
      }
    }
  }

  size_t next_block_size(size_t upcoming_allocation_size) {
    size_t actual_block_size = std::max(upcoming_allocation_size, next_block_size_);
    next_block_size_ = std::min<size_t>(actual_block_size * options_.growth_factor,
                                        options_.max_block_size);
    return actual_block_size;
  }

  memory_resource *upstream_;
  FreeList free_list_;
  LockType lock_;
  pool_options options_;
  size_t next_block_size_ = 0;

  struct UpstreamBlock {
    void *ptr;
    size_t bytes, alignment;
  };

  SmallVector<UpstreamBlock, 16> blocks_;
  using lock_guard = std::lock_guard<LockType>;
  using unique_lock = std::unique_lock<LockType>;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_POOL_RESOURCE_H_
