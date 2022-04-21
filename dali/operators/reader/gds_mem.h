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
#include <memory>
#include <mutex>
#include <unordered_set>
#include "dali/core/mm/memory.h"
#include "dali/core/int_literals.h"
#include "dali/core/spinlock.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {

size_t GetGDSChunkSize();

class GDSAllocator {
 public:
  explicit GDSAllocator(int device_id = -1);

  mm::memory_resource<mm::memory_kind::device> *resource() const {
    return rsrc_.get();
  }

  static GDSAllocator &instance(int device_id = -1);

 private:
  std::shared_ptr<mm::memory_resource<mm::memory_kind::device>> rsrc_;
};


inline std::shared_ptr<uint8_t> gds_alloc(size_t bytes) {
    return mm::alloc_raw_shared<uint8_t>(GDSAllocator::instance().resource(), bytes);
}

inline mm::uptr<uint8_t> gds_alloc_unique(size_t bytes) {
    return mm::alloc_raw_unique<uint8_t>(GDSAllocator::instance().resource(), bytes);
}

class GDSStagingBuffer {
 public:
  GDSStagingBuffer(GDSStagingBuffer &&other) noexcept {
    base = other.base;
    other.base = nullptr;
  }
  GDSStagingBuffer &&operator=(GDSStagingBuffer &&other) noexcept(false) {
    if (base)
      throw std::logic_error("Cannot overwrite a non-null staging buffer pointer.");
    base = other.base;
    other.base = nullptr;
  }

  void* at(ptrdiff_t offset) const { return static_cast<char *>(base) + offset; }
 private:
  explicit GDSStagingBuffer(void *base) : base(base) {}
  void *release() {
    void *ptr = base;
    base = nullptr;
    return ptr;
  }
  ~GDSStagingBuffer() noexcept(false) {
    if (base)
      throw std::logic_error("A staging buffer must be returned to the staging engine");
  }

  friend class GDSStagingEngine;
  void *base = nullptr;
};

class GDSStagingEngine {
 public:
  GDSStagingEngine(int device_id = -1, int max_buffers = 64, int commit_after = 32);
  void set_stream(cudaStream_t stream);
  GDSStagingBuffer get_staging_buffer(void *hint = nullptr);

  void return_buffer(GDSStagingBuffer &&buf);

  void copy_to_client(void *client_buffer, size_t nbytes,
                      GDSStagingBuffer &&staging_buffer, ptrdiff_t offset = 0);
  void commit();

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

  using StagingBuffer = mm::uptr<uint8_t>;

  kernels::ScatterGatherGPU copy_engine_;
  std::vector<StagingBuffer> blocks_;
  std::unordered_set<void*> ready_buffers_;

  std::vector<void *> unscheduled_;
  std::vector<void *> scheduled_;
  CUDAEvent ready_;

};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_GDS_MEM_H_
