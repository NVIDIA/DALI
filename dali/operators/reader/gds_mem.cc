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

#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "dali/operators/reader/gds_mem.h"
#include "dali/core/spinlock.h"
#include "dali/core/mm/pool_resource.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/composite_resource.h"
#include "dali/util/cufile_helper.h"
#include <cufile.h>

namespace dali {

static size_t GetGDSChunkSizeEnv() {
  if (char *env = getenv("DALI_GDS_CHUNK_SIZE")) {
    size_t s = atoll(env);
    DALI_ENFORCE(!is_pow2(s), "GDS chunk size must be a power of 2");
    DALI_ENFORCE(s >= 4096 && s <= (16 << 10), "GDS chunk size must be between 4 KiB and 16 MiB");
    return s;
  } else {
    return 2 << 20;  // default - 2 MiB
  }
}

size_t GetGDSChunkSize() {
  static size_t chunk_size = GetGDSChunkSizeEnv();
  return chunk_size;
}

void RegisterChunks(void *start, int chunks, size_t chunk_size) {
  //cuFileRegister
}

void UnregisterChunks(void *start, int chunks, size_t chunk_size) {
}

class GDSRegisteredResource : public mm::memory_resource<mm::memory_kind::device> {
 public:
  GDSRegisteredResource(int device_id) {
    cufile_driver_ = cufile::CUFileDriverHandle::Get(device_id);
  }
 private:

  void adjust_params(size_t &size, size_t &alignment) {
    if (alignment < chunk_size_) {
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
    std::unique_lock<spinlock> ul(map_lock_);
    auto it = allocs_.find(ptr);
    if (it == allocs_.end())
      throw std::invalid_argument("The pointer was not allocated by this resource "
                                  "or has been deleted.");
    alloc_info alloc = it->second;
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
  std::shared_ptr<cufile::CUFileDriverHandle> cufile_driver_;
};

class GDSMemoryResource : public mm::memory_resource<mm::memory_kind::device> {
 public:
  GDSMemoryResource(int device_id) : upstream_(device_id), pool_(&upstream_, pool_opts()) {
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
  mm::pool_resource_base<mm::memory_kind::device, mm::coalescing_free_tree, spinlock> pool_;
};

GDSAllocator::GDSAllocator(int device_id) {
  if (device_id < 0)
    CUDA_CALL(cudaGetDevice(&device_id));
  // Currently, GPUDirect Storage can work only with memory allocated with cudaMalloc and
  // cuMemAlloc. Since DALI is transitioning to CUDA Virtual Memory Management for memory
  // allocation, we need a special allocator that's compatible with GDS.
  rsrc_ = std::make_shared<GDSMemoryResource>(device_id);
}

struct GDSAllocatorInstance {
  GDSAllocator &get(int device_id) {
    if (alloc_)
      return *alloc_;
    std::lock_guard<std::mutex> g(mtx_);
    if (alloc_)
      return *alloc_;
    alloc_ = std::make_unique<GDSAllocator>(device_id);
    return *alloc_;
  }
  std::unique_ptr<GDSAllocator> alloc_;
  std::mutex mtx_;
};

GDSAllocator &GDSAllocator::instance(int device) {
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

}  // namespace dali
