// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_MM_ASYNC_POOL_H_
#define DALI_CORE_MM_ASYNC_POOL_H_

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>
#include "dali/core/mm/pool_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream.h"

namespace dali {
namespace mm {

template <memory_kind kind, class FreeList, class LockType, class Upstream = memory_resource<kind>>
class async_pool_base : public stream_aware_memory_resource<kind> {
 public:
  explicit async_pool_base(Upstream *upstream) : global_pool_(upstream) {
  }
  ~async_pool_base() {
    try {
      synchronize();
    } catch (const CUDAError &e) {
      if ((e.is_rt_api() && e.rt_error() != cudaErrorCudartUnloading) ||
          (e.is_drv_api() && e.drv_error() != CUDA_ERROR_DEINITIALIZED))
        std::terminate();
    }
  }

  /**
   * @brief Waits until all pending frees are finished.
   */
  void synchronize() {
    {
      std::lock_guard<LockType> guard(lock_);
      for (auto &kv : stream_free_) {
        if (!kv.second.free_list.head)
          continue;
        if (!sync_stream_)
          sync_stream_ = CUDAStream::Create(true);
        CUDA_DTOR_CALL(cudaStreamWaitEvent(sync_stream_, kv.second.free_list.head->event, 0));
      }
    }
    CUDA_DTOR_CALL(cudaStreamSynchronize(sync_stream_));
  }

 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    adjust_size_and_alignment(bytes, alignment);
    std::lock_guard<LockType> guard(lock_);
    return allocate_from_global_pool(bytes, alignment);
  }

  void do_deallocate(void *mem, size_t bytes, size_t alignment) override {
    adjust_size_and_alignment(bytes, alignment);
    std::lock_guard<LockType> guard(lock_);
    char *ptr = static_cast<char *>(mem);
    restore_original_params(ptr, bytes, alignment);
    return global_pool_.deallocate(ptr, bytes, alignment);
  }

  /**
   * @brief Tries to recycle per-stream free memory or allocates from global pool.
   *
   * There are four cases:
   * (1) Attempt to allocate from stream-specific free list - a smallest suitable block is used,
   *     padding is applied, if necessary, and recorded.
   * (2) If (1) fails and there are no pending per-stream free operations, the requested memory
   *     is allocated from the global pool.
   * (3) If (1) fails and there are free pending per-stream operations, an attempt is made to get
   *     memory from the free list of the underlying global pool, without using upstream resource.
   *     If this allocation succeeds, the memory allocated this way is returned.
   * (4) If (3) fails, the pending per-stream frees are collected to the global pool and then
   *     an allocation is made from the global pool.
   *
   * Per-stream frees are not coalesced - therefore, even if the total memory freed on a stream
   * may be sufficient
   */
  void *do_allocate_async(size_t bytes, size_t alignment, stream_view stream) override {
    adjust_size_and_alignment(bytes, alignment);
    std::lock_guard<LockType> guard(lock_);
    auto it = stream_free_.find(stream.value());
    void *ptr;
    if (it != stream_free_.end()) {
      ptr = try_allocate(it->second, bytes, alignment);
      if (ptr)
        return ptr;
    }
    return allocate_from_global_pool(bytes, alignment);
  }

  void *allocate_from_global_pool(size_t bytes, size_t alignment) {
    void *ptr;
    if (num_pending_frees_ == 0) {
      ptr = global_pool_.allocate(bytes, alignment);
      return ptr;
    }
    ptr = global_pool_.try_allocate_from_free(bytes, alignment);
    if (ptr)
      return ptr;
    for (auto &kv : stream_free_)
      free_pending(kv.second);
    ptr = global_pool_.allocate(bytes, alignment);
    return ptr;
  }

  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view stream) override {
    adjust_size_and_alignment(bytes, alignment);
    std::lock_guard<LockType> guard(lock_);
    char *ptr = static_cast<char*>(mem);
    restore_original_params(ptr, bytes, alignment);
    deallocate_async_impl(stream_free_[stream.value()], ptr, bytes, alignment, stream.value());
  }

  static void adjust_size_and_alignment(size_t &size, size_t &alignment) {
    if (size == 0)
      return;
    int l = ilog2(size);
    size_t min_align = (1 << (l >> 1));  // 2^(l/2)
    if (min_align > 256)
      min_align = 256;
    if (min_align > alignment)
      alignment = min_align;
    size = align_up(size, alignment);
  }

  /// @brief Information about a pending `free` operation
  struct pending_free {
    char           *addr = nullptr;
    size_t          bytes = 0;
    size_t          alignment = alignof(std::max_align_t);
    CUDAEvent       event;
    bool            is_ready = false;
    pending_free   *prev = nullptr, *next = nullptr;

    bool ready() {
      if (is_ready)
        return true;
      cudaError_t e = cudaEventQuery(event);
      switch (e) {
        case cudaSuccess:
          return is_ready = true;
        case cudaErrorNotReady:
          return false;
        case cudaErrorCudartUnloading:
          cudaGetLastError();
          return true;
        default:
          cudaGetLastError();
          throw CUDAError(e);
      }
    }
  };

  struct PendingFreeList {
    pending_free *head = nullptr, *tail = nullptr;
  };

  /// @brief Non-coalescing free blocks deallocated on a stream
  struct PerStreamFreeBlocks {
    using size_pending = std::pair<size_t, pending_free *>;

    PerStreamFreeBlocks() = default;
    PerStreamFreeBlocks(const PerStreamFreeBlocks &) = delete;

    /// @brief A map that associates block size with the free block; used for best-fit allocation
    detail::pooled_set<size_pending, true> by_size;

    /**
     * @brief List of free blocks, stored in deallocation order, head being most recent.
     *
     * The items in the list are kept in the order there were deallocated. Since we're using
     * stream-ordered deallocation and all list items were deallocated on the same stream, this
     * list has a well defined transition point from pending to complete deallocations - this
     * is exploited when returning complete deallocations to the global pool.
     */
    PendingFreeList free_list;
  };

  /**
   * @brief Try to allocate a block from per-stream free memory
   *
   * If the allocation fails, the function returns nullptr.
   */
  void *try_allocate(PerStreamFreeBlocks &from, size_t bytes, size_t alignment) {
    int max_padding = std::min<size_t>(std::max<size_t>(bytes / 16, 16), (1 << 20));
    const int remainder_alignment = 16;
    const int min_split_remainder = 16;
    for (auto it = from.by_size.lower_bound({ bytes, nullptr }); it != from.by_size.end(); ++it) {
      size_t block_size = it->first;
      pending_free *f = it->second;
      char *base = f->addr;
      char *aligned = detail::align_ptr(base, alignment);
      size_t front_padding = aligned - base;
      assert(static_cast<ptrdiff_t>(front_padding) >= 0);
      // NOTE: block_size - front_padding >= size  can overflow and fail - meh, unsigned size_t
      if (block_size >= bytes + front_padding) {
        if (!supports_splitting && block_size - bytes - front_padding > max_padding)
          return nullptr;  // no point in continuing - the map is sorted

        from.by_size.erase(it);
        char *block_end = base + block_size;
        char *remainder = detail::align_ptr(aligned + bytes, remainder_alignment);
        bool split = supports_splitting && remainder + min_split_remainder < block_end;
        size_t split_size = block_size;
        if (split) {
          // Adjust the pending free `f` so that it contains only what remains after
          // the block was split.
          size_t remainder_size = block_end - remainder;
          f->addr = remainder;
          f->bytes = remainder_size;
          f->alignment = remainder_alignment;
          from.by_size.insert({ remainder_size, f });
          // The block taken out from the free list is now reduced
          split_size = remainder - base;
        } else {
          remove_pending_free(from, f, false);
        }
        if (split_size != bytes) {
          padded_[aligned] = { split_size,
                               static_cast<int>(front_padding),
                               static_cast<int>(alignment) };
        }
        return aligned;
      }
    }
    return nullptr;
  }

  /**
   * @brief Searches per-stream free blocks to find the most recently freed one.
   */
  pending_free *find_first_ready(PerStreamFreeBlocks &free) {
    SmallVector<pending_free *, 128> pending;
    int step = 1;
    pending_free *f = free.free_list.head;
    while (f) {
      if (f->ready())
        break;
      pending.clear();
      for (int i = 0; i < step; i++) {
        f = f->next;
        if (!f)
          break;
        pending.push_back(f);
      }
    }
    if (pending.empty()) {
      if (f) {
        assert(f->ready());
        assert(!f->prev || !f->ready());
      }
      return f;
    }
    auto it = std::partition_point(pending.begin(), pending.end(), [&](pending_free *f) {
      return !f->ready();
    });
    if (it == pending.end()) {
      assert(!free.free_list.tail || !free.free_list.tail->ready());
      return nullptr;
    }
    f = *it;
    assert(f->ready());
    return f;
  }

  void free_pending(PerStreamFreeBlocks &free) {
    auto *f = find_first_ready(free);
    while (f) {
      global_pool_.deallocate(f->addr, f->bytes, f->alignment);
      f = remove_pending_free(free, f);
    }
  }

  void deallocate_async_impl(PerStreamFreeBlocks &free, char *ptr, size_t bytes, size_t alignment,
                             cudaStream_t stream) {
    auto *pending = add_pending_free(free.free_list, ptr, bytes, alignment, stream);
    try {
      free.by_size.insert({bytes, pending});
    } catch (...) {
      remove_pending_free(free.free_list, pending);
      throw;
    }
  }

  /**
   * @brief Restores original block parameters (address, size and alignment) of a block
   *        that was used for allocating a smaller block, possibly with stricter alignment.
   */
  void restore_original_params(char *&p, size_t &bytes, size_t &alignment) {
    auto it = padded_.find(p);
    if (it != padded_.end()) {
      if (it->second.front_padding + bytes > it->second.bytes) {
        throw std::invalid_argument("The deallocated memory points to a block that's smaller than "
          "the size being freed. Check the size of the memory region being freed.");
      }
      p -= it->second.front_padding;
      bytes = it->second.bytes;
      alignment = it->second.alignment;
      padded_.erase(it);
    }
  }

  auto *add_pending_free(PendingFreeList &free, char *base, size_t bytes, size_t alignment,
                         cudaStream_t stream) {
    pending_free *f = FreeDescAlloc::allocate(1);
    f = new (f)pending_free();
    f->addr = base;
    f->bytes = bytes;
    f->alignment = alignment;
    f->prev = nullptr;
    f->next = free.head;
    if (f->next)
      f->next->prev = f;
    free.head = f;
    if (!free.tail) free.tail = f;
    f->event = event_pool_.Get();
    cudaEventRecord(f->event, stream);
    num_pending_frees_++;
    return f;
  }

  pending_free *remove_pending_free(PerStreamFreeBlocks &free, pending_free *f,
                                    bool remove_by_size = true) {
    if (remove_by_size)
      free.by_size.erase({ f->bytes, f });
    return remove_pending_free(free.free_list, f);
  }

  pending_free *remove_pending_free(PendingFreeList &free, pending_free *f) {
    event_pool_.Put(std::move(f->event));
    auto *prev = f->prev;
    auto *next = f->next;
    if (free.head == f)
      free.head = next;
    if (free.tail == f)
      free.tail = prev;
    if (prev) prev->next = next;
    if (next) next->prev = prev;
    *f = {};
#if 0  // check list consistency
    assert(!free.head || !free.head->prev);
    assert(!free.tail || !free.tail->next);
    for (auto *x = free.head; x; x = x->next) {
      assert(!x->next || x->next->prev == x);
      assert(!x->prev || x->prev->next == x);
      assert(x != f);
      assert(x->next || x == free.tail);
    }
#endif
    f->~pending_free();
    FreeDescAlloc::deallocate(f, 1);
    num_pending_frees_--;
    return next;
  }


  struct padded_block {
    size_t bytes;
    int front_padding;
    int alignment;
  };

  detail::pooled_map<char *, padded_block, true> padded_;

  std::unordered_map<cudaStream_t, PerStreamFreeBlocks> stream_free_;
  int num_pending_frees_ = 0;

  using FreeDescAlloc = detail::object_pool_allocator<pending_free>;

  LockType lock_;

  CUDAEventPool event_pool_;
  CUDAStream sync_stream_;

  static constexpr bool supports_splitting = detail::can_merge<FreeList>::value;
  pool_resource_base<kind, any_context, FreeList, detail::dummy_lock> global_pool_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_ASYNC_POOL_H_
