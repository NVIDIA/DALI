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

#include "dali/operators/decoder/nvjpeg/nvjpeg_memory.h"
#include <nvjpeg.h>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/mm/memory.h"
#include "dali/core/spinlock.h"

namespace dali {

namespace nvjpeg_memory {

template <typename MemoryKind>
struct MallocLikeWrapper {
  MallocLikeWrapper() : mr_(mm::ShareDefaultResource<MemoryKind>()) {}
  explicit MallocLikeWrapper(shared_ptr<mm::default_memory_resource_t<MemoryKind>> mr) : mr_(mr) {}
  MallocLikeWrapper(const MallocLikeWrapper &) = delete;

  void *Allocate(size_t bytes, size_t alignment) {
    mm::uptr<char *> uptr;
    uptr = mm::alloc_raw_unique<char *>(mr_.get(), bytes, alignment);;
    auto del = std::move(uptr.get_deleter());
    std::lock_guard<decltype(lock_)> guard(lock_);
    outstanding_allocs_[uptr.get()] = std::move(del);
    return uptr.release();
  }

  static void FreeAll() {
    std::lock_guard<decltype(lock_)> guard(lock_);
    for (auto &ptr_del : outstanding_allocs_) {
      mm::uptr<void> uptr(ptr_del.first, std::move(ptr_del.second));
      uptr.reset();  // delete
    }
  }

  static void Free(void *ptr) {
    if (!ptr)
      return;
    auto it = outstanding_allocs_.find(ptr);
    if (it == outstanding_allocs_.end())
      throw std::invalid_argument("The pointer passed to `Free` was not "
                                  "allocated by this allocator.");
    mm::uptr<void> uptr(it->first, std::move(it->second));
    uptr.reset();  // delete
    std::lock_guard<decltype(lock_)> guard(lock_);
    outstanding_allocs_.erase(it);
  }

  std::shared_ptr<mm::default_memory_resource_t<MemoryKind>> mr_;
  static spinlock lock_;
  static std::unordered_map<void *, mm::Deleter> outstanding_allocs_;
};

template <typename MemoryKind>
MallocLikeWrapper<MemoryKind> &GetAllocator() {
  static MallocLikeWrapper<MemoryKind> wrapper;
  return wrapper;
}

struct AllocInfo {
  mm::Deleter deleter;
  std::thread::id thread_id;
};

std::map<void*, AllocInfo> alloc_info_;
std::shared_timed_mutex alloc_info_mutex_;

void SetEnableMemStats(bool enabled) {
}

void PrintMemStats() {
}


template <typename MemoryKind>
void AddBuffer(std::thread::id thread_id, size_t size) {
}

template void AddBuffer<mm::memory_kind::pinned>(std::thread::id thread_id, size_t size);
template void AddBuffer<mm::memory_kind::device>(std::thread::id thread_id, size_t size);

void DeleteAllBuffers(std::thread::id thread_id) {
}

template <typename MemoryKind>
static int NVJpegNew(void **ptr, size_t size) {
  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }
  // this function should not throw, but return a proper result
  try {
    *ptr = GetAllocator<MemoryKind>().Allocate(size, 256);
    return *ptr != nullptr ? cudaSuccess : cudaErrorMemoryAllocation;
  } catch (const std::bad_alloc &) {
    *ptr = nullptr;
    return cudaErrorMemoryAllocation;
  } catch (const CUDAError &e) {
    return e.is_rt_api() ? e.rt_error() : cudaErrorUnknown;
  } catch (...) {
    *ptr = nullptr;
    return cudaErrorUnknown;
  }
}


static int DeviceNew(void **ptr, size_t size) {
  return NVJpegNew<mm::memory_kind::device>(ptr, size);
}

static int PinnedNew(void **ptr, size_t size, unsigned int flags) {
  return NVJpegNew<mm::memory_kind::pinned>(ptr, size);
}

template <typename MemoryKind>
static int NVJpegFree(void *mem) {
  try {
    MallocLikeWrapper<MemoryKind>::Free(mem);
    return cudaSuccess;
  } catch (std::invalid_argument &) {
    return cudaErrorInvalidValue;
  }
}

static int DeviceFree(void *mem) {
  return NVJpegFree<mm::memory_kind::device>(mem);
}

static int PinnedFree(void *mem) {
  return NVJpegFree<mm::memory_kind::pinned>(mem);
}

nvjpegDevAllocator_t GetDeviceAllocator() {
  nvjpegDevAllocator_t allocator;
  allocator.dev_malloc = &DeviceNew;
  allocator.dev_free = &DeviceFree;
  return allocator;
}

nvjpegPinnedAllocator_t GetPinnedAllocator() {
  nvjpegPinnedAllocator_t allocator;
  allocator.pinned_malloc = &PinnedNew;
  allocator.pinned_free = &PinnedFree;
  return allocator;
}

#if NVJPEG2K_ENABLED
nvjpeg2kDeviceAllocator_t GetDeviceAllocatorNvJpeg2k() {
  nvjpeg2kDeviceAllocator_t allocator;
  allocator.device_malloc = &DeviceNew;
  allocator.device_free = &DeviceFree;
  return allocator;
}

nvjpeg2kPinnedAllocator_t GetPinnedAllocatorNvJpeg2k() {
  nvjpeg2kPinnedAllocator_t allocator;
  allocator.pinned_malloc = &PinnedNew;
  allocator.pinned_free = &PinnedFree;
  return allocator;
}
#endif  // NVJPEG2K_ENABLED

}  // namespace nvjpeg_memory

}  // namespace dali
