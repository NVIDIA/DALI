// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/core/mm/pool_resource.h"
#include "dali/core/mm/with_upstream.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/mm/detail/stream_id_hint.h"
#include "dali/core/small_vector.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_error.h"
#include "dali/core/device_guard.h"

#ifndef DEBUG_ASYNC_POOL_CHECK_CONSISTENCY
#define DEBUG_ASYNC_POOL_CHECK_CONSISTENCY 0
#endif

namespace dali {
namespace mm {

template <typename Kind,
    typename GlobalPool = pool_resource<Kind, coalescing_free_tree, spinlock>,
    typename LockType = std::mutex,
    typename Upstream = memory_resource<Kind>>
class async_pool_resource : public async_memory_resource<Kind>,
                            public pool_resource_base<Kind>,
                            public with_upstream<Kind> {
 public:
  /**
   * @param upstream       Upstream resource, used by the global pool
   * @param avoid_upstream If true, synchronize with outstanding deallocations before
   *                       using upstream.
   */
  template <typename P = GlobalPool,
            typename = std::enable_if_t<std::is_constructible<P, Upstream*, pool_options>::value>>
  explicit async_pool_resource(Upstream *upstream, bool avoid_upstream = true)
  : global_pool_(upstream, global_pool_options()), avoid_upstream_(avoid_upstream) {
    CUDAEventPool::instance();
  }

  /**
   * @param upstream       Upstream resource, used by the global pool
   * @param avoid_upstream If true, synchronize with outstanding deallocations before
   *                       using upstream.
   */
  template <typename... PoolArgs,
            typename P = GlobalPool,
            typename = std::enable_if_t<std::is_constructible<P, PoolArgs...>::value>>
  explicit async_pool_resource(PoolArgs&&...args)
  : global_pool_(std::forward<PoolArgs>(args)...), avoid_upstream_(false) {
    CUDAEventPool::instance();
  }

  ~async_pool_resource() {
    try {
      synchronize();
    } catch (const CUDAError &e) {
      if (!e.is_unloading())
        std::terminate();
    }
    for (auto &kv : stream_free_) {
      PerStreamFreeBlocks &free_blocks = kv.second;
      std::lock_guard<std::mutex> guard(lock_);

      // find_first_ready doesn't throw on shutdown - and we want to terminate on other errors
      assert(!stream_id_hint::is_unambiguous() ||
             find_first_ready(free_blocks) == free_blocks.free_list.head);

      for (auto *f = free_blocks.free_list.head; f; ) {
        assert(f->ready());
        try {
          global_pool_.deallocate(f->addr, f->bytes, f->alignment);
        } catch (const CUDAError &e) {
          if (!e.is_unloading())
            std::terminate();
        }
        f = remove_pending_free(free_blocks, f, false);
      }
      assert(free_blocks.free_list.head == nullptr);
      assert(free_blocks.free_list.tail == nullptr);
      free_blocks.by_size.clear();
    }
  }

  /**
   * @brief Waits until all pending frees are finished.
   */
  void synchronize() {
    synchronize_impl(true);
  }

  /**
   * @brief Releases unused upstream memory
   *
   * Releases any ready per-stream blocks to the global pool and
   * calls `release_unused` on it.
   */
  void release_unused() override {
    std::lock_guard<std::mutex> guard(lock_);
    synchronize_impl(false);
    for (auto &kv : stream_free_)
      free_ready(kv.second);
    global_pool_.release_unused();
  }

  void *try_allocate_from_free(size_t size, size_t alignment) override {
    std::lock_guard<std::mutex> guard(lock_);
    return global_pool_.try_allocate_from_free(size, alignment);
  }

  GlobalPool *upstream() const override {
    // ugly WAR - the global pool may be the "upstream" we really care about here
    return const_cast<GlobalPool *>(&global_pool_);
  }

 private:
  void synchronize_impl(bool lock) {
    {
      std::unique_lock<std::mutex> ulock(lock_, std::defer_lock);
      if (lock)
        ulock.lock();
      for (auto &kv : stream_free_) {
        if (!kv.second.free_list.head)
          continue;

        ContextScope scope(kv.second.free_list.head->ctx);
        int dev = 0;
        CUDA_DTOR_CALL(cudaGetDevice(&dev));
        CUDA_DTOR_CALL(cudaStreamWaitEvent(GetSyncStream(dev), kv.second.free_list.head->event, 0));
      }
    }
    for (int dev = 0; dev < static_cast<int>(sync_streams_.size()); dev++) {
      if (sync_streams_[dev])
        CUDA_DTOR_CALL(cudaStreamSynchronize(sync_streams_[dev]));
    }
  }

  void *do_allocate(size_t bytes, size_t alignment) override {
    adjust_size_and_alignment(bytes, alignment, true);
    std::lock_guard<LockType> guard(lock_);
    return allocate_from_global_pool(bytes, alignment);
  }

  void do_deallocate(void *mem, size_t bytes, size_t alignment) override {
    if (!mem || !bytes)
      return;
    adjust_size_and_alignment(bytes, alignment, false);
    std::lock_guard<LockType> guard(lock_);
    char *ptr = static_cast<char *>(mem);
    pop_block_padding(ptr, bytes, alignment);
    global_pool_.deallocate(ptr, bytes, alignment);
  }

  /**
   * @brief Tries to recycle per-stream free memory or allocates from the global pool.
   *
   * There are two main cases:
   * (1) Attempt to allocate from stream-specific free list - a smallest suitable block is used,
   *     padding is applied, if necessary, and recorded.
   * (2) Otherwise, the requested memory is allocated from global pool.
   *
   * Per-stream frees are not coalesced - therefore, even if the total memory freed on a stream
   * may be sufficient, there may be no satisfactory block.
   */
  void *do_allocate_async(size_t bytes, size_t alignment, stream_view stream) override {
    if (!bytes)
      return nullptr;
    adjust_size_and_alignment(bytes, alignment, true);
    std::lock_guard<LockType> guard(lock_);
    auto it = stream_free_.find(stream_id_hint::from_handle(stream.get()));
    void *ptr;
    if (it != stream_free_.end()) {
      ptr = try_allocate(it->second, bytes, alignment, stream);
      if (ptr)
        return ptr;
    }
    return allocate_from_global_pool(bytes, alignment);
  }

  /**
   * @brief Allocates from the global pool, possibly releasing per-stream memory to the global pool.
   */
  void *allocate_from_global_pool(size_t bytes, size_t alignment) {
    void *ptr;
    if (num_pending_frees_ == 0) {
      // There are no pending per-stream frees - there's no hope of reclaiming anything.
      ptr = global_pool_.allocate(bytes, alignment);
      return ptr;
    }
    // Try to allocate from the global pool, without using upstream
    ptr = global_pool_.try_allocate_from_free(bytes, alignment);
    if (ptr)
      return ptr;
    // Try to reclaim some memory from completed pending frees.
    for (auto &kv : stream_free_)
      free_ready(kv.second);
    if (avoid_upstream_ && num_pending_frees_ > 0) {
      // AVOIDING UPSTREAM
      // Try to allocate from the global pool again...
      ptr = global_pool_.try_allocate_from_free(bytes, alignment);
      if (ptr)
        return ptr;
      // Synchronize - this will wait for pending frees to complete.
      synchronize_impl(false);
      for (auto &kv : stream_free_)
        free_ready(kv.second);
    }
    // Finally, allocate from the global pool, this time allowing fallback to the upstream.
    ptr = global_pool_.allocate(bytes, alignment);
    return ptr;
  }

  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view stream) override {
    if (!mem || !bytes)
      return;
    std::cerr << "deallocate " << bytes << " @" << mem << " stream = "
              << (void*)stream.get() << std::endl;
    adjust_size_and_alignment(bytes, alignment, false);

    // Record an event outside of the mutex - it may take considerable time
    CUcontext ctx;
    CUDA_CALL(cuStreamGetCtx(stream.get(), &ctx));
    ContextScope scope(ctx);
    CUDAEvent event = CUDAEventPool::instance().Get();
    CUDA_CALL(cudaEventRecord(event, stream.get()));

    std::lock_guard<LockType> guard(lock_);
    char *ptr = static_cast<char*>(mem);
    pop_block_padding(ptr, bytes, alignment);
    auto stream_id = stream_id_hint::from_handle(stream.get());
    deallocate_async_impl(stream_free_[stream_id], ptr, bytes, alignment, ctx, std::move(event));
  }

  /**
   * @brief Increases minimum alignment based on size and quantizes the size to a multiple
   *        of the alignment.
   *
   * If the alignment requirements are low compared to the size, the alignment requirement is
   * increased, which improves reusability of the block. Further, the block size is aligned
   * to a next multiple of the (new) alignment, reducing the need for block splitting in case
   * of multiple allocations of similarly-sized blocks.
   */
  static void adjust_size_and_alignment(size_t &size, size_t &alignment, bool check) {
    if (size == 0)
      return;
    int log2size = ilog2(size);
    size_t min_align = (1 << (log2size >> 1));  // 2^(log2size/2)
    if (min_align > 256)
      min_align = 256;
    if (min_align > alignment)
      alignment = min_align;
    size_t aligned = align_up(size, alignment);
    if (check && aligned < size)
      throw std::bad_alloc();
    size = aligned;
  }

  /// @brief Information about a pending `free` operation
  struct pending_free {
    char           *addr = nullptr;
    size_t          bytes = 0;
    size_t          alignment = alignof(std::max_align_t);
    CUDAEvent       event;
    CUcontext       ctx = nullptr;
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
  void *try_allocate(PerStreamFreeBlocks &from,
                     size_t bytes,
                     size_t alignment,
                     stream_view stream) {
    // This value is only used when not splitting - it limits how much memory
    // can be wasted for padding - the allowed padding is 1/16 of the allocated size,
    // clamped to between 16 bytes and 1 MiB.
    unsigned max_padding = std::min<size_t>(std::max<size_t>(bytes / 16, 16), (1 << 20));
    const unsigned remainder_alignment = 16;
    const unsigned min_split_remainder = 16;
    for (auto it = from.by_size.lower_bound({ bytes, nullptr }); it != from.by_size.end(); ++it) {
      /*
      base - start of the existing free block
      aligned - base of the block, aligned to meet requested alignment
      remainder - base of the new free block when splitting
      block_end - pointer to the first byte not in the block

      block_size - size of the block before splitting
      bytes - the requested allocation size
      remainder_size - the size of the new free block, left after splitting
      front_padding - padding applied to the base to meet requested alignment
      rem_pad - padding applied to the remainder to meet remainder_alignment

      |<-------------------------- block_size ----------------------------->|
      |<------------ split_size ------------------->|                       |
      v                                             v                       v
      ^                   ^                         ^                       ^
      |<- front_padding ->|<- bytes -> <- rem_pad-> |<-- remainder_size --> |
      |                   |                         |                       |
      |___ base           |__aligned                |__remainder            |__ block_end


      Threshold values:
      min_split_remainder - minimum remainder we care about to add to split list (as opposed to
                            simply wasting it as padding)
      max_padding         - maximum allowed wasted size when _not splitting_
      */

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
        size_t orig_alignment = f->alignment;
        size_t split_size = block_size;
        // If stream_id is ambiguous, we might have blocks from other streams mixed in
        if (!stream_id_hint::is_unambiguous()) {
          // This check is not necessary for correctness but skipping it resulted in a big
          // performance hit in allocator performance tests.
          if (!f->ready())
            CUDA_CALL(cudaStreamWaitEvent(stream.get(), f->event));
        }
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
        if (split_size != bytes || alignment != orig_alignment) {
          padded_[aligned] = { split_size,
                               static_cast<int>(front_padding),
                               static_cast<int>(orig_alignment) };
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
    // Without a reliable stream id we might have
    // blocks from different streams mixed up - there's no
    // strict ordering and find_first makes no sense.
    assert(stream_id_hint::is_unambiguous());
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
      step += step;
    }
    if (pending.empty()) {
      if (f) {
        assert(f->is_ready);
        assert(!f->prev || !f->is_ready);
      }
      return f;
    }
    auto it = std::partition_point(pending.begin(), pending.end(), [&](pending_free *f) {
      return !f->ready();
    });
    if (it == pending.end()) {
      assert(!free.free_list.tail || !free.free_list.tail->is_ready);
      return nullptr;
    }
    f = *it;
    assert(f->ready());
    assert(!f->prev || !f->prev->is_ready);
    return f;
  }

  /**
   * @brief Returns the memory from completed deallocations to the global pool.
   */
  void free_ready(PerStreamFreeBlocks &free) {
    /*if (stream_id_hint::is_unambiguous()) {
      // strict order available - find the newest ready, all older entries must be ready too
      auto *f = find_first_ready(free);
      while (f) {
        global_pool_.deallocate(f->addr, f->bytes, f->alignment);
        f = remove_pending_free(free, f);
      }
    } else*/ {
      // no strict order - go over elements and remove ones that are ready
      for (auto *f = free.free_list.head; f; ) {
        std::cerr << "Checking pending free block: " << f->bytes << " @" << (void*)f->addr << "...";
        if (f->ready()) {
          std::cerr << "free\n";
          global_pool_.deallocate(f->addr, f->bytes, f->alignment);
          f = remove_pending_free(free, f);
        } else {
          std::cerr << "busy\n";
          f = f->next;
        }
      }
    }
  }

  void deallocate_async_impl(PerStreamFreeBlocks &free, char *ptr, size_t bytes, size_t alignment,
                             CUcontext ctx, CUDAEvent &&event) {
    auto *pending = add_pending_free(free.free_list, ptr, bytes, alignment, ctx, std::move(event));
    try {
      free.by_size.insert({bytes, pending});
    } catch (...) {
      remove_pending_free(free.free_list, pending);
      throw;
    }
  }

  /**
   * @brief Applies block padding and removes padding info.
   *
   * Applies block padding (adjusts address, size and alignment) of a block that was
   * truncated/split for allocating a smaller block (possibly with stricter alignment)
   * and removes padding from `padded_` map.
   */
  void pop_block_padding(char *&p, size_t &bytes, size_t &alignment) {
    auto it = padded_.find(p);
    if (it != padded_.end()) {
      assert(it->second.front_padding + bytes <= it->second.bytes &&
        "The deallocated memory points to a block that's smaller than "
        "the size being freed. Check the size of the memory region being freed.");
      p -= it->second.front_padding;
      bytes = it->second.bytes;
      alignment = it->second.alignment;
      padded_.erase(it);
    }
  }

  auto *add_pending_free(PendingFreeList &free, char *base, size_t bytes, size_t alignment,
                         CUcontext ctx, CUDAEvent &&event) {
    if (!cuInitChecked())
      throw std::runtime_error("Cannot load CUDA driver API library");
    pending_free *f = FreeDescAlloc::allocate(1);
    f = new (f)pending_free();
    f->ctx = ctx;
    f->event = std::move(event);
    f->addr = base;
    f->bytes = bytes;
    f->alignment = alignment;
    f->prev = nullptr;
    f->next = free.head;
    if (f->next)
      f->next->prev = f;
    free.head = f;
    if (!free.tail) free.tail = f;
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
    ContextScope scope(f->ctx);
    CUDAEventPool::instance().Put(std::move(f->event));
    auto *prev = f->prev;
    auto *next = f->next;
    if (free.head == f)
      free.head = next;
    if (free.tail == f)
      free.tail = prev;
    if (prev) prev->next = next;
    if (next) next->prev = prev;
    *f = {};
#if DEBUG_ASYNC_POOL_CHECK_CONSISTENCY
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

  std::unordered_map<uint64_t, PerStreamFreeBlocks> stream_free_;

  using FreeDescAlloc = detail::object_pool_allocator<pending_free>;

  LockType lock_;
  vector<CUDAStream> sync_streams_;
  CUDAStream &GetSyncStream(int device_id) {
    int ndev = sync_streams_.size();
    if (sync_streams_.empty()) {
      CUDA_CALL(cudaGetDeviceCount(&ndev));
      sync_streams_.resize(ndev);
    }
    assert(device_id >= 0 && device_id < ndev);
    if (!sync_streams_[device_id])
      sync_streams_[device_id] = CUDAStream::Create(true, device_id);
    return sync_streams_[device_id];
  }

  /**
   * @brief Sets a new context for the lifetime of the object
   *
   * Unlike DeviceGuard, which focuses on restoring the old context upon destruction,
   * this object is optimized to reduce the number of API calls and doesn't restore
   * the old context if the new context and current context are the same at construction.
   */
  struct ContextScope {
    explicit ContextScope(CUcontext new_ctx) {
      CUDA_CALL(cuCtxGetCurrent(&old_ctx));
      if (old_ctx == new_ctx) {
        old_ctx = nullptr;
      } else {
        CUDA_CALL(cuCtxSetCurrent(new_ctx));
      }
    }
    ~ContextScope() {
      if (old_ctx) {
        CUDA_DTOR_CALL(cuCtxSetCurrent(old_ctx));
      }
    }

   private:
    CUcontext old_ctx;
  };

  /**
   * @brief Indicates whether the global pool supports splitting
   *
   * In general, `memory_resource` requires that the a pointer being deallocated
   * was returned from a previous allocation on the same resource, with the same size.
   * However, a specific implementation of a memory resource (i.e. pool_resource with
   * certain FreeList types) can concatenate the deallocated memory segments.
   * This property is used for partial recycling of stream-bound free blocks - we might
   * reuse a part of the block on the same stream and return the rest to the global pool, with
   * the hope of reducing the number of calls to upstream and overall memory consumption.
   */
  static constexpr bool supports_splitting = detail::can_merge<GlobalPool>::value;

  static constexpr pool_options global_pool_options() {
    return default_pool_opts<Kind>();
  }

  GlobalPool global_pool_;

  int num_pending_frees_ = 0;
  bool avoid_upstream_ = true;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_ASYNC_POOL_H_
