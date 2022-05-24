// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_GDS_MEM_H_
#define DALI_OPERATORS_READER_GDS_MEM_H_

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>
#include "dali/core/mm/memory.h"
#include "dali/core/int_literals.h"
#include "dali/core/spinlock.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {
namespace gds {

static constexpr size_t kGDSAlignment = 4096;
DLL_PUBLIC size_t GetGDSChunkSize();

class DLL_PUBLIC GDSAllocator {
 public:
  GDSAllocator();

  mm::memory_resource<mm::memory_kind::device> *resource() const {
    return rsrc_.get();
  }

  static std::shared_ptr<GDSAllocator> get(int device_id = -1);

  /**
   * @brief Allocates a block of size `bytes`, usable with GDS
   *
   * The memory allocated is registered with GDS and suitably aligned, so it can be used
   * with GDS wihtout additional staging/copying.
   *
   * The block must not outlive the GDSAllocator which allocated it.
   *
   * @param bytes The size of the block to allocate
   */
  inline std::shared_ptr<uint8_t> alloc_shared(size_t bytes) {
      return mm::alloc_raw_shared<uint8_t>(resource(), bytes);
  }

  /**
   * @brief Allocates a block of size `bytes`, usable with GDS
   *
   * The memory allocated is registered with GDS and suitably aligned, so it can be used
   * with GDS wihtout additional staging/copying.
   *
   * The block must not outlive the GDSAllocator which allocated it.
   *
   * @param bytes The size of the block to allocate
   */
  inline mm::uptr<uint8_t> alloc_unique(size_t bytes) {
      return mm::alloc_raw_unique<uint8_t>(resource(), bytes);
  }

 private:
  std::unique_ptr<mm::memory_resource<mm::memory_kind::device>> rsrc_;
};


class GDSStagingBuffer {
 public:
  GDSStagingBuffer(GDSStagingBuffer &&other) noexcept {
    base = other.base;
    other.base = nullptr;
  }
  GDSStagingBuffer &operator=(GDSStagingBuffer &&other) {
    assert(!base && "Cannot overwrite a non-null staging buffer pointer.");
    base = other.base;
    other.base = nullptr;
    return *this;
  }

  void* at(ptrdiff_t offset) const { return static_cast<char *>(base) + offset; }

  ~GDSStagingBuffer() {
    assert(!base && "A staging buffer must be returned to the staging engine");
  }

 private:
  explicit GDSStagingBuffer(void *base) : base(base) {}
  void *release() {
    void *ptr = base;
    base = nullptr;
    return ptr;
  }

  friend class GDSStagingEngine;
  void *base = nullptr;
};

/**
 * @brief Manages copies from GDS-registered staging buffers to user-provided target buffers.
 */
class DLL_PUBLIC GDSStagingEngine {
 public:
  explicit GDSStagingEngine(int device_id = -1, int max_buffers = 64, int commit_after = 32);

  /**
   * @brief Sets the stream used for copying from staging buffers to client buffers.
   */
  void set_stream(cudaStream_t stream);

  /**
   * @brief Obtains a single-chunk buffer that's aligned to meet GDS requirements.
   *
   * If the pool of buffers is depleted, the function waits for the pending commits to complete.
   *
   * @param hint Preferred starting address of the buffer - useful when trying to obtain sequential
   *             buffers for coalescing.
   */
  GDSStagingBuffer get_staging_buffer(void *hint = nullptr);

  /**
   * @brief Enqueues a copy from a staging buffer to a client-provided destination buffer.
   *
   * Enqueues a copy from a staging buffer to a client buffer. The copy is enqueued, but not
   * scheduled for execution until either a call to `commit` or the number of copies reached
   * the value specified in `commit_after`.
   *
   *
   * @param client_buffer   the destination buffer, to which the contents of the staging buffer
   *                        will be copied.
   * @param nbytes          the amount of data to copy
   * @param staging_buffer  the source buffer; when the copy completes, it is returned to the pool
   * @param staging_offset  the offset in the staging (source) buffer to copy from
   */
  void copy_to_client(void *client_buffer, size_t nbytes,
                      GDSStagingBuffer &&staging_buffer, ptrdiff_t staging_offset = 0);

  /**
   * @brief Immediately returns an unused buffer to the staging engine's buffer pool.
   */
  void return_unused(GDSStagingBuffer &&buf);

  /**
   * @brief Schedules the enqueued copies on the stream.
   */
  void commit();

  size_t chunk_size() const { return chunk_size_; }

 private:
  int allocate_buffers();
  void wait_buffers(std::unique_lock<std::mutex> &lock);
  void return_ready();
  void commit_no_lock();

  std::mutex lock_;
  int device_id_ = 0;
  int commit_after_ = 32;
  int max_buffers_ = 64;
  size_t chunk_size_ = GetGDSChunkSize();
  cudaStream_t stream_ = 0;
  std::shared_ptr<GDSAllocator> allocator_;

  using StagingBuffer = mm::uptr<uint8_t>;

  kernels::ScatterGatherGPU copy_engine_;
  std::vector<StagingBuffer> blocks_;
  std::unordered_set<void*> ready_buffers_;

  std::vector<void *> unscheduled_;
  std::vector<void *> scheduled_;
  CUDAEvent ready_;
};

}  // namespace gds
}  // namespace dali

#endif  // DALI_OPERATORS_READER_GDS_MEM_H_
