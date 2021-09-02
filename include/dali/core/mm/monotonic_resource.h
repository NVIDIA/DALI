// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_MONOTONIC_RESOURCE_H_
#define DALI_CORE_MM_MONOTONIC_RESOURCE_H_

#include <cassert>
#include <algorithm>
#include <utility>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/util.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace mm {

/**
 * @brief Monotonic resource which returns memory from a preallocated buffer.
 *
 * Monotonic buffer resources don't require manual deallocation of the returned pointers.
 * The memory is freed in bulk when the underlying buffer is freed.
 */
template <typename Kind, typename Context = any_context>
class monotonic_buffer_resource : public memory_resource<Kind, Context> {
 public:
  monotonic_buffer_resource() = default;
  monotonic_buffer_resource(void *memory, size_t bytes)
  : curr_(static_cast<char*>(memory)), limit_(static_cast<char*>(memory) + bytes) {}

  monotonic_buffer_resource(monotonic_buffer_resource &&other) {
    swap(other);
  }

  monotonic_buffer_resource &operator=(monotonic_buffer_resource &&other) {
    curr_ = nullptr;
    limit_ = nullptr;
    swap(other);
    return *this;
  }

  void swap(monotonic_buffer_resource &other) {
    std::swap(curr_, other.curr_);
    std::swap(limit_, other.limit_);
  }

  size_t avail(size_t alignment) const {
    return limit_ - detail::align_ptr(curr_, alignment);
  }

 protected:
  void *do_allocate(size_t bytes, size_t alignment) override {
    char *ret = detail::align_ptr(curr_, alignment);
    if (ret + bytes > limit_)
      throw std::bad_alloc();
    curr_ = ret + bytes;
    return ret;
  }

  // don't deallocate at all
  void do_deallocate(void *data, size_t bytes, size_t alignment) override {
  }

  char *curr_ = nullptr, *limit_ = nullptr;
};

template <typename Kind, typename Context = any_context,
          bool host_impl = detail::is_host_accessible<Kind>>
class monotonic_memory_resource;

/**
 * @brief Monotonic resource which allocates memory from an upstream resource, storing
 *        the metadata in the same memory blocks as the allocated buffers.
 *
 * Monotonic resources don't require manual deallocation of memory.
 * The lifetime of a monotonic resource is limited and all memory will be deallocated in bulk
 * when the resource is destroyed.
 */
template <typename Kind, typename Context>
class monotonic_memory_resource<Kind, Context, true> : public memory_resource<Kind, Context> {
 public:
  explicit monotonic_memory_resource(memory_resource<Kind, Context> *upstream,
                                     size_t next_block_size = 1024)
  : upstream_(upstream), next_block_size_(next_block_size) {}

  monotonic_memory_resource() = default;

  monotonic_memory_resource(monotonic_memory_resource &&other) {
    swap(other);
  }

  monotonic_memory_resource &operator=(monotonic_memory_resource &&other) {
    free_all();
    swap(other);
    return *this;
  }

  void swap(monotonic_memory_resource &other) {
    std::swap(curr_, other.curr_);
    std::swap(limit_, other.limit_);
    std::swap(upstream_, other.upstream_);
    std::swap(next_block_size_, other.next_block_size_);
    std::swap(curr_block_, other.curr_block_);
  }

  ~monotonic_memory_resource() {
    free_all();
  }

  /**
   * @brief Releases all memory blocks that were taken from upstream
   */
  void free_all() {
    while (curr_block_) {
      assert(curr_block_->sentinel == sentinel_value && "Memory corruption detected");
      auto *prev = curr_block_->prev;
      auto *base = reinterpret_cast<char*>(curr_block_) - curr_block_->usable_size;
      size_t alloc_size = curr_block_->usable_size + sizeof(block_info);
      upstream_->deallocate(base, alloc_size, curr_block_->alignment);
      curr_block_ = prev;
    }
  }

 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    char *ret = detail::align_ptr(curr_, alignment);
    if (ret + bytes > limit_) {
      while (next_block_size_ < bytes + sizeof(block_info))
        next_block_size_ += next_block_size_;
      alignment = std::max(alignment, alignof(block_info));
      curr_ = static_cast<char*>(upstream_->allocate(next_block_size_, alignment));
      limit_ = curr_ + next_block_size_ - sizeof(block_info);
      block_info *prev = curr_block_;
      if (prev) {
        assert(prev->sentinel == sentinel_value && "Memory corruption detected");
      }
      curr_block_ = reinterpret_cast<block_info*>(limit_);
      curr_block_->sentinel = sentinel_value;
      curr_block_->prev = prev;
      curr_block_->usable_size = next_block_size_ - sizeof(block_info);
      curr_block_->alignment = alignment;
      next_block_size_ += next_block_size_;

      ret = curr_;
    }

    curr_ = ret + bytes;
    return ret;
  }

  // don't deallocate at all
  void do_deallocate(void *data, size_t bytes, size_t alignment) override {
  }

  Context do_get_context() const noexcept override {
    return upstream_->get_context();
  }

  static constexpr size_t sentinel_value = detail::sentinel_value<size_t>::value;

  char *curr_ = nullptr, *limit_ = nullptr;

  memory_resource<Kind, Context> *upstream_;
  size_t next_block_size_;
  struct block_info {
    size_t sentinel;
    block_info *prev;
    size_t usable_size;
    size_t alignment;
  };
  block_info *curr_block_ = nullptr;
};

/**
 * @brief Monotonic resource that keeps separate host memory to manage upstream blocks.
 *
 * Use this resource when the upstream memory cannot be accessed by the main processor for the
 * purpose of storing the memory management structures.
 *
 * Monotonic resources don't require manual deallocation of memory.
 * The lifetime of a monotonic resource is limited and all memory will be deallocated in bulk
 * when the resource is destroyed.
 */
template <typename Kind, typename Context>
class monotonic_memory_resource<Kind, Context, false> : public memory_resource<Kind, Context> {
 public:
  explicit monotonic_memory_resource(memory_resource<Kind, Context> *upstream,
                                     size_t next_block_size = 1024)
  : upstream_(upstream), next_block_size_(next_block_size) {}

  monotonic_memory_resource() = default;

  monotonic_memory_resource(monotonic_memory_resource &&other) {
    swap(other);
  }

  monotonic_memory_resource &operator=(monotonic_memory_resource &&other) {
    free_all();
    swap(other);
    return *this;
  }

  void swap(monotonic_memory_resource &other) {
    std::swap(curr_, other.curr_);
    std::swap(limit_, other.limit_);
    std::swap(upstream_, other.upstream_);
    std::swap(next_block_size_, other.next_block_size_);
    std::swap(blocks_, other.blocks_);
  }

  ~monotonic_memory_resource() {
    free_all();
  }

  /**
   * @brief Releases all memory blocks that were taken from upstream
   */
  void free_all() {
    for (int i = blocks_.size() - 1; i >= 0; i--) {
      auto &blk = blocks_[i];
      upstream_->deallocate(blk.base, blk.size, blk.alignment);
    }
    blocks_.clear();
  }


 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    char *ret = detail::align_ptr(curr_, alignment);
    if (ret + bytes > limit_) {
      while (next_block_size_ < bytes)
        next_block_size_ += next_block_size_;

      alignment = std::max(alignof(std::max_align_t), alignment);
      curr_ = static_cast<char*>(upstream_->allocate(next_block_size_, alignment));
      limit_ = curr_ + next_block_size_;
      blocks_.push_back({ curr_, next_block_size_, alignment });
      ret = curr_;
      next_block_size_ += next_block_size_;
    }

    curr_ = ret + bytes;
    return ret;
  }

  // don't deallocate at all
  void do_deallocate(void *data, size_t bytes, size_t alignment) override {
  }

  Context do_get_context() const noexcept override {
    return upstream_->get_context();
  }

  char *curr_ = nullptr, *limit_ = nullptr;

  memory_resource<Kind, Context> *upstream_;
  size_t next_block_size_;
  struct upstream_block {
    void *base;
    size_t size;
    size_t alignment;
  };
  SmallVector<upstream_block, 8> blocks_;
};

using monotonic_host_resource = monotonic_memory_resource<memory_kind::host>;

template <typename Context = any_context>
using monotonic_device_resource = monotonic_memory_resource<memory_kind::device, Context>;

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MONOTONIC_RESOURCE_H_
