// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "dali/operators/reader/gds_mem.h"
#include "dali/util/cufile_helper.h"
#include "dali/core/math_util.h"
#include "dali/core/spinlock.h"
#include "dali/core/mm/pool_resource.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/composite_resource.h"

namespace dali {
namespace gds {

static size_t GetGDSChunkSizeEnv() {
  char *env = getenv("DALI_GDS_CHUNK_SIZE");
  int len = 0;
  if (env && (len = strlen(env))) {
    for (int i = 0; i < len; i++) {
      bool valid = std::isdigit(env[i]) || (i == len - 1 && (env[i] == 'k' || env[i] == 'M'));
      if (!valid) {
        DALI_FAIL(make_string(
          "DALI_GDS_CHUNK_SIZE must be a number, optionally followed by 'k' or 'M', got: ",
          env));
      }
    }
    size_t s = atoll(env);
    if (env[len-1] == 'k')
      s <<= 10;
    else if (env[len-1] == 'M')
      s <<= 20;
    DALI_ENFORCE(is_pow2(s), make_string("GDS chunk size must be a power of two, got ", s));
    DALI_ENFORCE(s >= kGDSAlignment && s <= (16 << 20),
      make_string("GDS chunk size must be a power of two between ",
                  kGDSAlignment, " and 16 M, got: ", s));
    return s;
  } else {
    // not set or empty
    return 2 << 20;  // default - 2 MiB
  }
}

size_t GetGDSChunkSize() {
  static size_t chunk_size = GetGDSChunkSizeEnv();
  return chunk_size;
}

void RegisterChunks(void *start, int chunks, size_t chunk_size) {
  char *addr = static_cast<char *>(start);
  char *end = addr + chunks * chunk_size;
  for (; addr < end; addr += chunk_size) {
    CUDA_CALL(cuFileBufRegister(addr, chunk_size, 0));
  }
}

void UnregisterChunks(void *start, int chunks, size_t chunk_size) {
  char *addr = static_cast<char *>(start);
  char *end = addr + chunks * chunk_size;
  for (; addr < end; addr += chunk_size) {
    CUDA_CALL(cuFileBufDeregister(addr));
  }
}

///////////////////////////////////////////////////////////////////////////////
// GDSRegisteredResource

class GDSRegisteredResource : public mm::memory_resource<mm::memory_kind::device> {
  cufile::CUFileDriverScope driver_scope_;

  void adjust_params(size_t &size, size_t &alignment) {
    if (alignment < chunk_size_) {
      // Both chunk size and alignment are powers of 2, so if chunk_size_ is larger than
      // the required alignment, it's also a multiple of it.
      alignment = chunk_size_;
    }
    size += alignment;  // we could go with alignment - 256, but that would risk having adjacent
                        // blocks which might get glued in the pool - and that would lead to
                        // failures in cudaMemcpy etc.
  }

  void *do_allocate(size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    char *raw = nullptr;
    CUDA_CALL(cudaMalloc(&raw, size));
    char *aligned = mm::detail::align_ptr(raw, alignment);
    size_t padding = aligned - raw;
    size_t chunks = (size - padding) / chunk_size_;
    assert(chunks * chunk_size_ >= size);
    RegisterChunks(aligned, chunks, chunk_size_);
    std::lock_guard<spinlock> g(map_lock_);
    allocs_[aligned] = { raw, size };
    return aligned;
  }

  void do_deallocate(void *ptr, size_t size, size_t alignment) override {
    adjust_params(size, alignment);
    std::unique_lock<spinlock> ul(map_lock_);
    auto it = allocs_.find(ptr);
    if (it == allocs_.end())
      throw std::invalid_argument("The pointer was not allocated by this resource "
                                  "or has been deleted.");
    alloc_info alloc = it->second;
    allocs_.erase(it);
    ul.unlock();
    assert(alloc.size == size);
    size_t padding = static_cast<char*>(ptr) - alloc.base_ptr;
    size_t chunks = (size - padding) / chunk_size_;
    UnregisterChunks(ptr, chunks, chunk_size_);
    CUDA_CALL(cudaFree(alloc.base_ptr));
  }

  size_t chunk_size_ = GetGDSChunkSize();
  spinlock map_lock_;
  struct alloc_info {
    char *base_ptr;
    size_t size;
  };
  std::map<void *, alloc_info> allocs_;
};

///////////////////////////////////////////////////////////////////////////////
// GDSMemoryPool

class GDSMemoryPool : public mm::memory_resource<mm::memory_kind::device> {
 public:
  GDSMemoryPool() : pool_(&upstream_, pool_opts()) {
  }
 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    alignment = std::max(alignment, GetGDSChunkSize());
    return pool_.allocate(bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    alignment = std::max(alignment, GetGDSChunkSize());
    return pool_.deallocate(ptr, bytes, alignment);
  }

  static mm::pool_options pool_opts() {
    auto opt = mm::default_pool_opts<mm::memory_kind::device>();
    opt.max_upstream_alignment = GetGDSChunkSize();
    return opt;
  }
  GDSRegisteredResource upstream_;
  mm::pool_resource<mm::memory_kind::device,
                         mm::coalescing_free_tree,
                         spinlock> pool_;
};

///////////////////////////////////////////////////////////////////////////////
// GDSAllocator

GDSAllocator::GDSAllocator() {
  // Currently, GPUDirect Storage can work only with memory allocated with cudaMalloc and
  // cuMemAlloc. Since DALI is transitioning to CUDA Virtual Memory Management for memory
  // allocation, we need a special allocator that's compatible with GDS.
  rsrc_ = std::make_unique<GDSMemoryPool>();
}

struct GDSAllocatorInstance {
  std::shared_ptr<GDSAllocator> get(int device_id) {
    std::shared_ptr<GDSAllocator> alloc = alloc_.lock();
    if (alloc)
      return alloc;
    std::lock_guard<std::mutex> g(mtx_);
    alloc = alloc_.lock();
    if (alloc)
      return alloc;
    alloc = std::make_shared<GDSAllocator>();
    alloc_ = alloc;
    return alloc;
  }
  std::weak_ptr<GDSAllocator> alloc_;
  std::mutex mtx_;
};

std::shared_ptr<GDSAllocator> GDSAllocator::get(int device) {
  static int ndevs = []() {
    int devs = 0;
    CUDA_CALL(cudaGetDeviceCount(&devs));
    return devs;
  }();
  static vector<GDSAllocatorInstance> instances(ndevs);
  if (device < 0)
    CUDA_CALL(cudaGetDevice(&device));
  assert(device >= 0 && device < ndevs);
  return instances[device].get(device);
}

///////////////////////////////////////////////////////////////////////////////
// GDSStagingEngine

GDSStagingEngine::GDSStagingEngine(int device_id, int max_buffers, int commit_after) {
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  device_id_ = device_id;
  max_buffers_ = max_buffers;
  commit_after_ = commit_after;

  allocator_ = GDSAllocator::get(device_id);
  ready_ = CUDAEventPool::instance().Get(device_id);
}

void GDSStagingEngine::set_stream(cudaStream_t stream) {
  this->stream_ = stream;
}

int GDSStagingEngine::allocate_buffers() {
  int prev = blocks_.size();
  int new_total = clamp(prev * 2, 2, max_buffers_);
  for (int i = prev; i < new_total; i++) {
    blocks_.push_back(allocator_->alloc_unique(chunk_size_));
    ready_buffers_.insert(blocks_.back().get());
  }
  return blocks_.size() - prev;
}

void GDSStagingEngine::wait_buffers(std::unique_lock<std::mutex> &lock) {
  if (scheduled_.empty() && !unscheduled_.empty())
    commit_no_lock();
  lock.unlock();
  CUDA_CALL(cudaEventSynchronize(ready_));
  lock.lock();
  for (void *ptr : scheduled_)
    ready_buffers_.insert(ptr);
  scheduled_.clear();
}

void GDSStagingEngine::return_unused(GDSStagingBuffer &&staging_buffer) {
  std::lock_guard g(lock_);
  ready_buffers_.insert(staging_buffer.release());
}

void GDSStagingEngine::copy_to_client(void *client_buffer, size_t nbytes,
                                      GDSStagingBuffer &&staging_buffer, ptrdiff_t offset) {
  std::lock_guard g(lock_);
  assert(staging_buffer.at(0) != nullptr && "Staging buffer already released.");
  copy_engine_.AddCopy(client_buffer, staging_buffer.at(offset), nbytes);
  unscheduled_.push_back(staging_buffer.release());
  if (unscheduled_.size() > static_cast<size_t>(commit_after_))
    commit_no_lock();
}

void GDSStagingEngine::return_ready() {
  cudaError_t ret = cudaEventQuery(ready_);
  if (ret == cudaErrorNotReady)
    return;
  CUDA_CALL(ret);
  for (void *ptr : scheduled_)
    ready_buffers_.insert(ptr);
  scheduled_.clear();
}

void GDSStagingEngine::commit() {
  std::lock_guard g(lock_);
  commit_no_lock();
}

void GDSStagingEngine::commit_no_lock() {
  copy_engine_.Run(stream_, true, kernels::ScatterGatherGPU::Method::Default,
                   cudaMemcpyDeviceToDevice);
  return_ready();
  if (scheduled_.empty()) {
    scheduled_.swap(unscheduled_);
  } else {
    scheduled_.insert(scheduled_.end(), unscheduled_.begin(), unscheduled_.end());
    unscheduled_.clear();
  }
  CUDA_CALL(cudaEventRecord(ready_, stream_));
}

GDSStagingBuffer GDSStagingEngine::get_staging_buffer(void *hint) {
  std::unique_lock ulock(lock_);
  for (;;) {
    auto it = ready_buffers_.find(hint);
    // try to get a buffer by hint
    if (it != ready_buffers_.end()) {
      void *ptr = *it;
      ready_buffers_.erase(it);
      return GDSStagingBuffer(ptr);
    }
    // hint failed - use any
    it = ready_buffers_.begin();
    if (it != ready_buffers_.end()) {
      void *ptr = *it;
      ready_buffers_.erase(it);
      return GDSStagingBuffer(ptr);
    }

    // no ready buffers? try to allocate more
    if (!allocate_buffers()) {
      // couldn't allocate? wait for some
      wait_buffers(ulock);
    }
  }
}

}  // namespace gds
}  // namespace dali
