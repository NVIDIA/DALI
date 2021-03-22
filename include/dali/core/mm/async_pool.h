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
  void *do_allocate(size_t bytes, size_t alignment) {

  }

  /// @brief Information about a pending `free` operation
  struct pending_free {
    char           *addr;
    size_t          size;
    size_t          alignment;
    CUDAEvent       event;
    pending_free   *prev, *next;
  };

  struct PendingFreeList {
    pending_free *head = nullptr, *tail = nullptr;
    int size = 0;
  };

  struct PerStreamFreeBlocks {
    using size_addr = std::pair<size_t, char *>;

    pooled_map<char *, pending_free*, true> by_addr_;
    pooled_set<size_addr, true> by_size;
    PendingFreeList free_list;
  };

  void *try_allocate(PerStreamFreeBlocks &from, size_t bytes, size_t alignment) {
    for (auto it = from.by_size.lower_bound({ size, nullptr }); it != from.by_size.end(); ++it) {
      size_t block_size = it->first;
      char *base = it->second;
      char *aligned = detail::align_ptr(base, alignment);
      size_t front_padding = aligned - base;
      assert(static_cast<ptrdiff_t>(front_padding) >= 0);
      // NOTE: block_size - front_padding >= size  can overflow and fail - meh, unsigned size_t
      if (block_size >= size + front_padding) {
        by_size_.erase(it);
        remove_pendinng_free(from, base, false);
        size_t back_padding = block_size - size - front_padding;
        assert(static_cast<ptrdiff_t>(back_padding) >= 0);
        if (front_padding || back_padding) {
          padded_[aligned] = { front_padding, back_padding };
        }
        return aligned;
      }
    }
    return nullptr;
  }

  pending_free *find_first_ready(PerStreamFreeBlocks &free) {
    SmallVector<pending_free *, 128> pending;
    int step = 1;
    pending_free *f = free.free_list.tail;
    while (f) {
     cudaError_t e = cudaEventQuery(f->event);
      if (!ready(f->event)) {
        pending.clear();
        for (int i = 0; i < step; i++) {
          f = f->prev;
          if (!f)
            break;
          pending.push_back(f);
        }
      } else if (e == cudaErrorCudartUnloading) {
        return nullptr;
      } else {
        throw CUDAError(e);
      }
    }
    if (pending.empty())
      return f;
    auto it = std::partition_point(pending.begin(), pending.end(), [&](pending_free *f) {
      return !ready(f->event);
    });
    if (it == pending.end())
      return nullptr;
    return &*it;
  }

  void free_pending(PerStreamFreeBlocks &free) {
    auto *f = find_first_ready(free);
  }

  void deallocate_impl(char *ptr, size_t size, size_t alignment, cudaStream_t stream, bool async) {
    if (async) {
      deallocate_async_impl(stream_free_[stream], ptr, size, stream);
    } else {
      global_pool_->deallocate(ptr, size, alignment);
    }
  }

  void deallocate_async_impl(PerStreamFreeBlock &free, char *ptr, size_t size,
                             cudaStream_t stream) {
    free.pending = add_pending_free(free.free_list, ptr, size, stream, host);
    free.by_size.insert({size, ptr});
    free.by_addr.insert(ptr, pending);
  }

  void restore_padding(char *&p, size_t &size) {
    auto *it = padded_.find(p)
    if (it != padded_.end()) {
      int front = it->second.first;
      int back  = it->second.second;
      p -= front;
      size += front + back;
      padded_.erase(it);
    }
  }

  auto *add_pending_free(PendingFreeList &free, char *base, size_t size, cudaStream_t stream) {
    pending_free *f = FreeDescAlloc::allocate(1);
    new (f)pending_free();
    f->prev = free.tail;
    free.tail = f;
    f->event = event_pool_.Get();
    cudaEventRecord(event, stream);
    return f;
  }

  void remove_pending_free(PerStreamFreeBlocks &free, char *base, bool remove_by_size = true) {
    auto it = by_addr_.find(base);
    assert(it != by_addr_.end());
    if (remove_by_size)
      by_size_.erase({ it->second.size, base });
    remove_pending_free(free.free_list, it->second);
    by_addr_.erase(it);
  }

  void remove_pending_free(PendingFreeList &free, char *base) {
    auto it = by_addr_.find(base);
    assert(it != by_addr_.end());
    pending_free *f = it->second;
    event_pool_.Put(std::move(f->event));
    auto *prev = free.prev;
    auto *next = free.next;
    if (free.head == f)
      free.head = next;
    if (free.tail == f)
      free.tail = prev;
    if (prev) prev->next = next;
    if (next) next->prev = prev;
    *f = {};
    f->~pending_free();
    FreeDescAlloc::deallocate(f, 1);
  }

  pooled_map<char *, std::pair<int, int>> padded_;

  std::unordered_map<cudaStream_t, PerStreamFreeBlocks> stream_free_;

  using FreeDescAlloc = object_pool_allocator<pending_free>;

  LockType lock_;

  CUDAEventPool event_pool_;

  pool_resource_base<kind, any_context, FreeList, detail::dummy_lock> global_pool_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_ASYNC_POOL_H_

