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

#ifndef DALI_CORE_MM_DETAIL_AUX_ALLOC_H_
#define DALI_CORE_MM_DETAIL_AUX_ALLOC_H_

#include <stdlib.h>
#include <malloc.h>
#include <algorithm>
#include <mutex>
#include "dali/core/spinlock.h"
#include "dali/core/mm/detail/util.h"

namespace dali {
namespace mm {
namespace detail {

/**
 * @brief An allocator that dispenses objects with fixed size and alignment.
 *
 * This allocator keeps underlying storage until explicitly purged or destroyed.
 * The objects are kept in an intrusive free list, where each item is allocated separately with
 * ordinary aligned malloc.
 *
 * It is safe to deallocate memory allocated by another instance of this class as long
 * as size and alignment requirements match and if the purge function links against the same
 * C runtime library.
 */
template <size_t size, size_t alignment, class LockObject = dali::spinlock>
struct fixed_size_allocator {
  ~fixed_size_allocator() {
    purge();
  }

  /**
   * @brief Deallocates the free list.
   */
  void purge() {
    lock_guard guard(lock_);
    while (free_list_) {
      Block *next = free_list_->next;
      free(free_list_);
      free_list_ = next;
    }
  }

  /**
   * @brief Gets an object from the free list, if avaliable, or allocates a new list entry and
   *        returns the pointer to the storage.
   *
   * @tparam T  type of the object to allocate. It must not exceed the size and alignment
   *            requirements of the storage.
   * @return A pointer to uninitialized memory suitable for storing an object of type T
   *
   * If the internal free list is not empty, the head of the list is removed
   * and the pointer to the storage block within the list entry is returned.
   * The pointer can be returned to the list by calling deallocate or freed directly with a call
   * to `free` on the returned pointer.
   */
  template <typename T>
  T *allocate() {
    static_assert(sizeof(T) <= size && alignof(T) <= alignment,
                  "Incompatible object size or alignment");
    {
      lock_guard guard(lock_);
      if (Block *blk = free_list_) {
        free_list_ = blk->next;
        blk->next = nullptr;
        return reinterpret_cast<T*>(&blk->storage);
      }
    }
    Block *blk = static_cast<Block*>(memalign(alignment, sizeof(Block)));
    blk->next = nullptr;
    return reinterpret_cast<T*>(&blk->storage);
  }

  /**
   * @brief Places the pointer ptr in the free list.
   *
   * @param ptr   a pointer to memory returned by a call to allocate in a fixed_size_allocator
   *              with the same size and alignment as this one.
   *
   * The caller relinquishes ownership of the pointer and it's in the free list for subsequent
   * reuse.
   */
  void deallocate(void *ptr) {
    lock_guard guard(lock_);
    Block *bptr = static_cast<Block*>(ptr);
    bptr->next = free_list_;
    free_list_ = bptr;
  }

  /**
   * @brief Returns a reference to a global instance of this allocator.
   */
  static fixed_size_allocator &instance() {
    static fixed_size_allocator inst;
    return inst;
  }

  /**
   * @brief Returns a per-thread instance of the allocator.
   *
   * The per-thread instance always uses a dummy lock, since there's no risk of concurrent
   * access in thread local storage.
   *
   * It is not safe to pass this reference to threads other than the calling one.
   */
  static fixed_size_allocator<size, alignment, dummy_lock> &thread_instance() {
    static thread_local fixed_size_allocator<size, alignment, dummy_lock> inst;
    return inst;
  }

  struct Block {
    std::aligned_storage_t<size, alignment> storage;
    Block *next;
  };
  Block *free_list_ = nullptr;
  LockObject lock_;
  using lock_guard = std::lock_guard<LockObject>;
};

/**
 * @brief A legacy C++98 allocator for fixed-size objects.
 *
 * This allocator is used for quickly managing a large number of fixed-size objects.
 * It CAN be used as an allocator for std::list, std::map and std::set.
 * It CANNOT be used for containers that allocate arrays of objects such as std::vector,
 * std::deque or unordered maps and sets.
 *
 * This is just a facade for an allocator parameterized with size and alignmnent - this
 * reduces the number of instances and allows for limited reusability of memory between different
 * object and container types.
 */
template <typename T, bool is_thread_local = false>
struct object_pool_allocator {
  template <typename U>
  constexpr bool operator==(object_pool_allocator<U, is_thread_local> other) const noexcept  {
    return sizeof(U) == sizeof(T) && alignof(U) == alignof(T);
  }
  template <typename U>
  constexpr bool operator!=(object_pool_allocator<U, is_thread_local> other) const noexcept {
    return !(*this == other);
  }

  template <typename U>
  struct rebind {
    using other = object_pool_allocator<U, is_thread_local>;
  };

  using BlockAllocator = fixed_size_allocator<sizeof(T), std::max(alignof(T), alignof(void*))>;

  using value_type = T;
  using pointer = T *;
  using reference = T &;
  static T *address(T &obj) { return &obj; }

  /**
   * @brief Allocates one object of type T
   *
   * @param n must be 1
   * @return A pointer to uninitialized memory suitable for storing an object of type T
   */
  static T *allocate(size_t n) {
    assert(n == 1);
    if (is_thread_local)
      return BlockAllocator::thread_instance().template allocate<T>();
    else
      return BlockAllocator::instance().template allocate<T>();
  }

  /**
   * @brief Deallocates an object of type T
   * @param ptr a pointer to memory returned by this allocator.
   * @param n   must be 1
   */
  static void deallocate(T *ptr, size_t n) {
    assert(n == 1);
    if (is_thread_local)
      return BlockAllocator::thread_instance().deallocate(ptr);
    else
      return BlockAllocator::instance().deallocate(ptr);
  }
};

}  // namespace detail
}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_AUX_ALLOC_H_
