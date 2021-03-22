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

#include <mutex>
#include <algorithm>
#include "dali/core/mm/pool_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"
#include "dali/core/cuda_event_pool.h"

namespace dali {
namespace mm {



template <memory_kind kind, class FreeList, class LockType, class Upstream = memory_resource<kind>>
class async_pool_base : public stream_aware_memory_resource<kind> {
 public:
  async_pool_base(Upstream *upstream) : global_pool_(upstream) {
  }
 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    std::lock_guard<LockType> guard(lock_);
    return global_pool_.allocate(bytes, alignment);
  }

  void do_deallocate(void *mem, size_t bytes, size_t alignment) override {
    std::lock_guard<LockType> guard(lock_);
    return global_pool_.deallocate(mem, bytes, alignment);
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
    std::lock_guard<LockType> guard(lock_);
    auto it = stream_free_.find(stream.value());
    void *ptr;
    if (it != stream_free_.end()) {
      ptr = try_allocate(it->second, bytes, alignment);
      if (ptr)
        return ptr;
    }
    if (num_pending_frees_ == 0)
      return global_pool_.allocate(bytes, alignment);
    ptr = global_pool_.try_allocate_from_free(bytes, alignment);
    if (ptr)
      return ptr;
    for (auto &kv : stream_free_)
      free_pending(kv.second);
    return global_pool_.allocate(bytes, alignment);
  }

  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view stream) override {
    std::lock_guard<LockType> guard(lock_);
    char *ptr = static_cast<char*>(mem);
    restore_original_params(ptr, bytes, alignment);
    deallocate_async_impl(stream_free_[stream.value()], ptr, bytes, alignment, stream.value());
  }

  /// @brief Information about a pending `free` operation
  struct pending_free {
    char           *addr;
    size_t          bytes;
    size_t          alignment;
    CUDAEvent       event;
    pending_free   *prev, *next;
  };

  struct PendingFreeList {
    pending_free *head = nullptr, *tail = nullptr;
  };

  struct PerStreamFreeBlocks {
    using size_addr = std::pair<size_t, char *>;

    detail::pooled_map<char *, pending_free*, true> by_addr;
    detail::pooled_set<size_addr, true> by_size;
    PendingFreeList free_list;
  };

  void *try_allocate(PerStreamFreeBlocks &from, size_t bytes, size_t alignment) {
    for (auto it = from.by_size.lower_bound({ bytes, nullptr }); it != from.by_size.end(); ++it) {
      size_t block_size = it->first;
      char *base = it->second;
      char *aligned = detail::align_ptr(base, alignment);
      size_t front_padding = aligned - base;
      assert(static_cast<ptrdiff_t>(front_padding) >= 0);
      // NOTE: block_size - front_padding >= size  can overflow and fail - meh, unsigned size_t
      if (block_size >= bytes + front_padding) {
        from.by_size.erase(it);
        remove_pending_free(from, base, false);
        size_t back_padding = block_size - bytes - front_padding;
        assert(static_cast<ptrdiff_t>(back_padding) >= 0);
        if (front_padding || back_padding) {
          padded_[aligned] = { block_size,
                               static_cast<int>(aligned-base),
                               static_cast<int>(alignment) };
        }
        return aligned;
      }
    }
    return nullptr;
  }

  static bool ready(cudaEvent_t event) {
    cudaError_t e = cudaEventQuery(event);
    switch (e) {
      case cudaSuccess:
        return true;
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

  pending_free *find_first_ready(PerStreamFreeBlocks &free) {
    SmallVector<pending_free *, 128> pending;
    int step = 1;
    pending_free *f = free.free_list.head;
    while (f) {
      if (ready(f->event))
        break;
      pending.clear();
      for (int i = 0; i < step; i++) {
        f = f->next;
        if (!f)
          break;
        pending.push_back(f);
      }
    }
    if (pending.empty())
      return f;
    auto it = std::partition_point(pending.begin(), pending.end(), [&](pending_free *f) {
      return !ready(f->event);
    });
    if (it == pending.end())
      return nullptr;
    return *it;
  }

  void free_pending(PerStreamFreeBlocks &free) {
    auto *f = find_first_ready(free);
    if (f == free.free_list.head) {
      free.free_list.head = nullptr;
      free.free_list.tail = nullptr;
    } else {
      free.free_list.tail = f->prev;
      f->prev->next = nullptr;
    }
    while (f) {
      global_pool_.deallocate(f->addr, f->bytes, f->alignment);
      f = remove_pending_free(free.free_list, f);
    }

  }

  void deallocate_async_impl(PerStreamFreeBlocks &free, char *ptr, size_t bytes, size_t alignment,
                             cudaStream_t stream) {
    auto *pending = add_pending_free(free.free_list, ptr, bytes, alignment, stream);
    free.by_size.insert({bytes, ptr});
    free.by_addr.insert({ptr, pending});
  }

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
    new (f)pending_free();
    f->addr = base;
    f->bytes = bytes;
    f->alignment = alignment;
    f->next = free.head;
    free.head = f;
    if (!free.tail) free.tail = f;
    f->event = event_pool_.Get();
    cudaEventRecord(f->event, stream);
    num_pending_frees_++;
    return f;
  }

  void remove_pending_free(PerStreamFreeBlocks &free, char *base, bool remove_by_size = true) {
    auto it = free.by_addr.find(base);
    assert(it != free.by_addr.end());
    if (remove_by_size)
      free.by_size.erase({ it->second->bytes, base });
    remove_pending_free(free.free_list, it->second);
    free.by_addr.erase(it);
  }

  void remove_pending_free(PendingFreeList &free, char *base) {
    auto it = free.by_addr.find(base);
    assert(it != free.by_addr.end());
    pending_free *f = it->second;
    remove_pending_free(free, f);
  }

  pending_free *remove_pending_free(PerStreamFreeBlocks &free, pending_free *f) {
    free.by_addr.erase(f->addr);
    free.by_size.erase(f->bytes);
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

  detail::pooled_map<char *, padded_block> padded_;

  std::unordered_map<cudaStream_t, PerStreamFreeBlocks> stream_free_;
  int num_pending_frees_ = 0;

  using FreeDescAlloc = detail::object_pool_allocator<pending_free>;

  LockType lock_;

  CUDAEventPool event_pool_;

  pool_resource_base<kind, any_context, FreeList, detail::dummy_lock> global_pool_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_ASYNC_POOL_H_

