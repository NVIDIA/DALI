// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <condition_variable>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/pool_resource_base.h"
#include "dali/core/mm/with_upstream.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"
#include "dali/core/device_guard.h"
#include "dali/core/util.h"

namespace dali {
namespace mm {

struct pool_options {
  /**
   * @brief Maximum block size
   *
   * Growth stops at this point; larger blocks are allocated only when allocate is called with
   * a larger memory requirements.
   */
  size_t max_block_size = -1_uz;  // no limit
  /// Minimum size of blocks requested from upstream
  size_t min_block_size = (1 << 12);
  /// The factor by which the allocation size grows until it reaches max_block_size
  float growth_factor = 2;
  /**
   * @brief Whether to try to allocate smaller blocks from upstream if default upcoming
   *        block is unavailable.
   */
  bool try_smaller_on_failure = true;
  /**
   * @brief Whether to try to return completely free blocks to the upstream when an allocation
   *        from upstream failed. This may effectively flush the pool.
   *
   * @remarks This option is ignored when `try_smaller_on_failure` is set to `false`.
   */
  bool return_to_upstream_on_failure = true;

  /**
   * @brief Minimum alignment used when allocating from the upstream resource.
   */
  size_t upstream_alignment = 256;

  /**
   * @brief Maximum supported alignment of the upstream resource.
   *
   * The upstream resource may not natively support alignments above certain value. If a larger
   * alignment is required, the next allocation from the upstream resource may need additional
   * padding to accommodate for the required (sub)allocation alignment.
   */
  size_t max_upstream_alignment = 256;
};

constexpr pool_options default_host_pool_opts() noexcept {
  return { (1 << 28), (1 << 12), 2.0f, true, true };
}

constexpr pool_options default_device_pool_opts() noexcept {
  return { (1_uz << 32), (1 << 20), 2.0f, true, true };
}

template <typename Kind>
constexpr pool_options default_pool_opts() noexcept {
  return default_device_pool_opts();
}

template <>
constexpr pool_options default_pool_opts<memory_kind::host>() noexcept {
  return default_host_pool_opts();
}

template <typename Kind, class FreeList, class LockType>
class pool_resource : public memory_resource<Kind>,
                      public pool_resource_base<Kind>,
                      public with_upstream<Kind> {
 public:
  explicit pool_resource(memory_resource<Kind> *upstream = nullptr,
                              const pool_options &opt = default_pool_opts<Kind>())
  : upstream_(upstream), options_(opt) {
     next_block_size_ = opt.min_block_size;
  }

  pool_resource(const pool_resource &) = delete;
  pool_resource(pool_resource &&) = delete;

  ~pool_resource() {
    free_all();
  }

  void free_all() {
    upstream_lock_guard uguard(upstream_lock_);
    lock_guard guard(lock_);
    for (auto &block : blocks_) {
      upstream_->deallocate(block.ptr, block.bytes, block.alignment);
    }
    blocks_.clear();
    free_list_.clear();
  }

  int device_ordinal() const noexcept {
    return device_ordinal_;
  }

  /**
   * @brief Tries to obtain a block from the internal free list.
   *
   * Allocates `bytes` memory from the free list. If a block that satisifies
   * the size or alignment requirements is not found, the function returns
   * nullptr withoug allocating from upstream.
   */
  void *try_allocate_from_free(size_t bytes, size_t alignment) override {
    if (!bytes)
      return nullptr;

    {
      lock_guard guard(lock_);
      return free_list_.get(bytes, alignment);
    }
  }

  memory_resource<Kind> *upstream() const override {
    return upstream_;
  }

  constexpr const pool_options &options() const noexcept {
    return options_;
  }

  /**
   * @brief Releases unused memory to the upstream resource
   *
   * If there are completely free upstream blocks, they are returned to upstream.
   * Partially used blocks remain allocated.
   */
  void release_unused() override {
    upstream_lock_guard uguard(upstream_lock_);
    release_unused_impl(true);
  }

 protected:
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (!bytes)
      return nullptr;

    if (void *ptr = try_allocate_from_free(bytes, alignment))
      return ptr;

    upstream_lock_guard uguard(upstream_lock_);
    // try again to avoid upstream allocation stampede
    if (void *ptr = try_allocate_from_free(bytes, alignment))
      return ptr;

    alignment = std::max(alignment, options_.upstream_alignment);
    size_t blk_size = bytes;

    char *block_start = static_cast<char*>(get_upstream_block(blk_size, bytes, alignment));
    assert(block_start);

    char *ret = detail::align_ptr(block_start, alignment);
    char *tail = ret + bytes;
    char *block_end = block_start + blk_size;
    assert(tail <= block_end);

    if (blk_size != bytes) {
      // we've allocated an oversized block - put the front & back padding in the free list
      lock_guard guard(lock_);
      if (ret != block_start)
        free_list_.put(block_start, ret - block_start);

      if (tail != block_end)
        free_list_.put(tail, block_end - tail);
    }
    return ret;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    if (static_cast<ssize_t>(bytes) < 0)
      throw std::bad_alloc();
    lock_guard guard(lock_);
    free_list_.put(ptr, bytes);
  }

  void *get_upstream_block(size_t &blk_size, size_t min_bytes, size_t alignment) {
    blk_size = next_block_size(min_bytes);
    if (alignment > options_.max_upstream_alignment) {
      // The upstream resource cannot guarantee the requested alignment - we must accommodate
      // for the extra padding, if necessary...
      if (blk_size < min_bytes + alignment - options_.max_upstream_alignment)
        blk_size = min_bytes + alignment - options_.max_upstream_alignment;
      // ...and relax the alignment requirement on the upstream allocation
      alignment = options_.max_upstream_alignment;
    }

    bool tried_return_to_upstream = false;
    void *new_block = nullptr;
    for (;;) {
      try {
        new_block = upstream_->allocate(blk_size, alignment);
        assert(new_block);
        break;
      } catch (const std::bad_alloc &) {
        if (!options_.try_smaller_on_failure)
          throw;
        if (blk_size == min_bytes) {
          // We've reached the minimum size and still got no memory from upstream
          // - try to free something.
          if (tried_return_to_upstream || !options_.return_to_upstream_on_failure)
            throw;
          if (blocks_.empty())  // nothing to free -> fail
            throw;
          // If there are some upstream blocks which are completely free
          // (the free list covers them completely), we can try to return them
          // to the upstream, with the hope that it will reorganize and succeed in
          // the subsequent allocation attempt.
          if (!release_unused_impl())
            throw;
          // mark that we've tried, so we can fail fast the next time
          tried_return_to_upstream = true;
        }
        blk_size = std::max(min_bytes, blk_size >> 1);

        // Shrink the next_block_size_, so that we don't try to allocate a big block
        // next time, because it would likely fail anyway.
        next_block_size_ = blk_size;
      }
    }
    try {
      blocks_.push_back({ new_block, blk_size, alignment });
    } catch (...) {
      upstream_->deallocate(new_block, blk_size, alignment);
      throw;
    }
    return new_block;
  }

  int release_unused_impl(bool shrink_next_block_size = false) {
    // go over blocks and find ones that are completely covered by free regions
    // - we can free such blocks.
    int blocks_freed = 0;
    SmallVector<bool, 32> removed;
    removed.resize(blocks_.size(), false);
    {
      lock_guard guard(lock_);
      for (int i = 0; i < static_cast<int>(blocks_.size()); i++) {
        UpstreamBlock blk = blocks_[i];
        removed[i] = free_list_.remove_if_in_list(blk.ptr, blk.bytes);
        if (removed[i])
          blocks_freed++;
      }
    }

    if (!blocks_freed)
      return 0;  // nothing to free

    // Go backwards, so we free in reverse order of allocation from upstream.
    for (int i = blocks_.size() - 1; i >= 0; i--) {
      if (removed[i]) {
        UpstreamBlock blk = blocks_[i];
        upstream_->deallocate(blk.ptr, blk.bytes, blk.alignment);
        blocks_.erase_at(i);
      }
    }
    if (shrink_next_block_size) {
      if (blocks_.empty()) {
        next_block_size_ = options_.min_block_size;
      } else {
        next_block_size_ = blocks_.back().bytes;
        // next_block_size() uses the previous value of next_block_size_ and modifies it
        // - this will behave now as if the last remaining block was the most recently allocated
        next_block_size(0);
      }
    }

    return blocks_freed;
  }

  size_t next_block_size(size_t upcoming_allocation_size) {
    size_t actual_block_size = std::max<size_t>(upcoming_allocation_size,
                                                next_block_size_ * options_.growth_factor);
    // Align the upstream block to reduce fragmentation.
    // The upstream resource (e.g. OS routine) may return blocks that have
    // coarse size granularity. This may result in fragmentation - the next
    // large block will be overaligned and we'll never see the padding.
    // Even though we might have received contiguous memory, we're not aware of that.
    // To reduce the probability of this happening, we align the size to 1/1024th
    // of the allocation size or 4kB (typical page size), whichever is larger.
    // This makes (at least sometimes) the large blocks to be seen as adjacent
    // and therefore enables coalescing in the free list.
    size_t alignment = 1uL << std::max((ilog2(actual_block_size) - 10), 12);
    actual_block_size = align_up(actual_block_size, alignment);
    next_block_size_ = std::min<size_t>(actual_block_size, options_.max_block_size);
    return actual_block_size;
  }

  memory_resource<Kind> *upstream_;
  FreeList free_list_;

  // locking order: upstream_lock_, lock_
  std::mutex upstream_lock_;
  LockType lock_;
  pool_options options_;
  size_t next_block_size_ = 0;
  int device_ordinal_ = -1;

  struct UpstreamBlock {
    void *ptr;
    size_t bytes, alignment;
  };

  SmallVector<UpstreamBlock, 16> blocks_;
  using lock_guard = std::lock_guard<LockType>;
  using unique_lock = std::unique_lock<LockType>;
  using upstream_lock_guard = std::lock_guard<std::mutex>;
};

namespace detail {

template <typename Kind, class FreeList, class LockType>
struct can_merge<pool_resource<Kind, FreeList, LockType>> : can_merge<FreeList> {};

}  // namespace detail

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_POOL_RESOURCE_H_
