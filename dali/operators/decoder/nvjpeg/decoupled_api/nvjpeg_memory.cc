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

#include "dali/operators/decoder/nvjpeg/decoupled_api/nvjpeg_memory.h"
#include <nvjpeg.h>
#include <shared_mutex>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include <memory>
#include "dali/core/cuda_error.h"
#include "dali/kernels/alloc.h"

namespace dali {

using kernels::memory::KernelUniquePtr;
using kernels::AllocType;
using kernels::memory::alloc_unique;
using kernels::memory::Allocate;

namespace nvjpeg_memory {

struct AllocInfo {
  AllocType alloc_type;
  size_t size;
  std::thread::id thread_id;
};

std::map<void*, AllocInfo> alloc_info_;
std::shared_timed_mutex alloc_info_mutex_;

struct Deleter {
  kernels::memory::Deleter deleter;
  explicit Deleter(AllocType alloc_type) : deleter(kernels::memory::GetDeleter(alloc_type)) {}

  inline void operator()(void *p) {
    deleter(p);
    std::unique_lock<std::shared_timed_mutex> lock(alloc_info_mutex_);
    alloc_info_.erase(p);
  }
};

using unique_ptr_t = std::unique_ptr<char, Deleter>;

struct Buffer {
  Buffer(unique_ptr_t unq_ptr, AllocType type, size_t sz)
      : ptr(std::move(unq_ptr)), alloc_type(type), size(sz) {}
  unique_ptr_t ptr;
  AllocType alloc_type;
  size_t size;
};

using MemoryPool = std::array<std::vector<Buffer>, static_cast<size_t>(AllocType::Count)>;
using BufferPool = std::map<std::thread::id, MemoryPool>;

BufferPool buffer_pool_;
std::shared_timed_mutex buffer_pool_mutex_;

struct MemoryStats {
  size_t nallocs = 0;
  size_t biggest_alloc = 0;
};

std::array<MemoryStats, static_cast<size_t>(AllocType::Count)> mem_stats_;
std::mutex mem_stats_mutex_;
std::atomic<bool> mem_stats_enabled_ = {true};

void SetEnableMemStats(bool enabled) {
  mem_stats_enabled_ = enabled;
}

void AddMemStats(AllocType alloc_type, size_t size) {
  if (mem_stats_enabled_) {
    std::lock_guard<std::mutex> lock(mem_stats_mutex_);
    auto &stats = mem_stats_[static_cast<size_t>(alloc_type)];
    stats.nallocs++;
    if (size > stats.biggest_alloc)
      stats.biggest_alloc = size;
  }
}

void PrintMemStats() {
  if (mem_stats_enabled_) {
    std::lock_guard<std::mutex> lock(mem_stats_mutex_);

    const char* log_filename = std::getenv("DALI_LOG_FILE");
    std::ofstream log_file;
    if (log_filename) log_file.open(log_filename);
    std::ostream& out = log_filename ? log_file : std::cout;
    out << std::dec;  // Don't want numbers printed as hex
    out << "#################### NVJPEG STATS ####################" << std::endl;
    auto &device_mem_stats = mem_stats_[static_cast<size_t>(AllocType::GPU)];
    out << "Device memory: " << device_mem_stats.nallocs
        << " allocations, largest = " << device_mem_stats.biggest_alloc << " bytes\n";
    auto &pinned_mem_stats = mem_stats_[static_cast<size_t>(AllocType::Pinned)];
    out << "Host (pinned) memory: " << pinned_mem_stats.nallocs
        << " allocations, largest = " << pinned_mem_stats.biggest_alloc << " bytes\n";
    out << "################## END NVJPEG STATS ##################" << std::endl;
  }
}

unique_ptr_t Allocate(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  auto ptr = kernels::memory::alloc_unique<char>(alloc_type, size);
  std::unique_lock<std::shared_timed_mutex> lock(alloc_info_mutex_);
  alloc_info_[ptr.get()] = {alloc_type, size, thread_id};
  return {ptr.release(), Deleter(alloc_type)};
}

void* GetBuffer(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  std::shared_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto it = buffer_pool_.find(thread_id);
  auto end_it =  buffer_pool_.end();
  lock.unlock();
  // only if pool exits for given thread_id search it, otherwise just allocate
  if (it != end_it) {
    auto &buffers = it->second[static_cast<size_t>(alloc_type)];
    auto best_fit = buffers.end();
    auto smallest = buffers.end();
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      if (smallest == buffers.end() || it->size < smallest->size) {
        smallest = it;
      }
      if (it->size >= size && (best_fit == buffers.end() || it->size < best_fit->size)) {
        best_fit = it;
      }
    }
    if (best_fit != buffers.end()) {
      std::swap(*best_fit, buffers.back());
      auto buffer = std::move(buffers.back());
      buffers.pop_back();
      return buffer.ptr.release();
    }
    if (smallest != buffers.end()) {
      std::swap(*smallest, buffers.back());
      buffers.pop_back();
    }
  }
  // Couldn't find a preallocated buffer, proceed to allocate
  AddMemStats(alloc_type, size);
  return Allocate(thread_id, alloc_type, size).release();
}

static int ReturnBufferToPool(void *raw_ptr) {
  std::shared_lock<std::shared_timed_mutex> info_lock(alloc_info_mutex_);
  auto info_it = alloc_info_.find(raw_ptr);
  assert(info_it != alloc_info_.end());
  auto info = info_it->second;
  info_lock.unlock();
  std::unique_ptr<char, Deleter> ptr(reinterpret_cast<char*>(raw_ptr), Deleter(info.alloc_type));
  std::shared_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto it = buffer_pool_.find(info.thread_id);
  auto end_it =  buffer_pool_.end();
  lock.unlock();
  MemoryPool *pool;
  if (it != end_it) {
    pool = &(it->second);
  } else {
    // if nothing has been preallocated create a pool for given thread_id
    std::unique_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
    pool = &(buffer_pool_[info.thread_id]);
  }
  auto &buffers = (*pool)[static_cast<size_t>(info.alloc_type)];
  buffers.emplace_back(std::move(ptr), info.alloc_type, info.size);
  return 0;
}

void AddBuffer(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  std::unique_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto& thread_buffer_pool = buffer_pool_[thread_id];
  lock.unlock();

  auto &buffers = thread_buffer_pool[static_cast<size_t>(alloc_type)];
  buffers.emplace_back(Allocate(thread_id, alloc_type, size), alloc_type, size);
  AddMemStats(alloc_type, size);
}

void DeleteAllBuffers(std::thread::id thread_id) {
  std::shared_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto it = buffer_pool_.find(thread_id);
  // no buffers have been preallocated/returned to the pool
  if (it == buffer_pool_.end()) {
    return;
  }
  lock.unlock();
  auto &buffers = it->second;
  for (auto &buffer : buffers)
    buffer.clear();
}

static int DeviceNew(void **ptr, size_t size) {
  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }
  // this function should not throw, but return a proper result
  try {
    *ptr = GetBuffer(std::this_thread::get_id(), AllocType::GPU, size);
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

static int PinnedNew(void **ptr, size_t size, unsigned int flags) {
  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }
  // this function should not throw, but return a proper result
  try {
    *ptr = GetBuffer(std::this_thread::get_id(), AllocType::Pinned, size);
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

nvjpegDevAllocator_t GetDeviceAllocator() {
  nvjpegDevAllocator_t allocator;
  allocator.dev_malloc = &DeviceNew;
  allocator.dev_free = &ReturnBufferToPool;
  return allocator;
}

nvjpegPinnedAllocator_t GetPinnedAllocator() {
  nvjpegPinnedAllocator_t allocator;
  allocator.pinned_malloc = &PinnedNew;
  allocator.pinned_free = &ReturnBufferToPool;
  return allocator;
}

#ifdef NVJPEG2K_ENABLED
nvjpeg2kDeviceAllocator_t GetDeviceAllocatorNvJpeg2k() {
  nvjpeg2kDeviceAllocator_t allocator;
  allocator.device_malloc = &DeviceNew;
  allocator.device_free = &ReturnBufferToPool;
  return allocator;
}

nvjpeg2kPinnedAllocator_t GetPinnedAllocatorNvJpeg2k() {
  nvjpeg2kPinnedAllocator_t allocator;
  allocator.pinned_malloc = &PinnedNew;
  allocator.pinned_free = &ReturnBufferToPool;
  return allocator;
}
#endif  // NVJPEG2K_ENABLED

}  // namespace nvjpeg_memory

}  // namespace dali
