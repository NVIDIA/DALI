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

#include "nvjpeg_memory.h"
#include <nvjpeg.h>
#include <atomic>
#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include "dali/kernels/alloc.h"
#include "dali/core/cuda_error.h"

namespace dali {

using kernels::memory::KernelUniquePtr;
using kernels::AllocType;
using kernels::memory::alloc_unique;
using kernels::memory::Allocate;

namespace nvjpeg_memory {

struct Buffer {
  Buffer(KernelUniquePtr<char> unq_ptr, AllocType type, size_t sz)
      : ptr(std::move(unq_ptr)), alloc_type(type), size(sz) {}
  KernelUniquePtr<char> ptr;
  AllocType alloc_type;
  size_t size;
};

using BufferPool = std::map<std::thread::id,
                            std::array<std::vector<Buffer>, static_cast<size_t>(AllocType::Count)>>;
std::shared_timed_mutex buffer_pool_mutex_;
BufferPool buffer_pool_;

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

void* GetBuffer(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  std::shared_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto it = buffer_pool_.find(thread_id);
  assert(it != buffer_pool_.end());
  lock.unlock();

  auto &buffers = it->second[static_cast<size_t>(alloc_type)];
  if (!buffers.empty()) {
    auto buf = std::move(buffers.back());
    buffers.pop_back();
    if (size <= buf.size)
      return buf.ptr.release();
  }
  // Couldn't find a preallocated buffer, proceed to allocate
  AddMemStats(alloc_type, size);
  return kernels::memory::Allocate(alloc_type, size);
}

void AddBuffer(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  std::unique_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto& thread_buffer_pool = buffer_pool_[thread_id];
  lock.unlock();

  auto &buffers = thread_buffer_pool[static_cast<size_t>(alloc_type)];
  buffers.emplace_back(kernels::memory::alloc_unique<char>(alloc_type, size), alloc_type, size);
  AddMemStats(alloc_type, size);
}

void DeleteAllBuffers(std::thread::id thread_id) {
  buffer_pool_.erase(thread_id);
}

static int DeviceNew(void **ptr, size_t size) {
  *ptr = GetBuffer(std::this_thread::get_id(), AllocType::GPU, size);
  return 0;
}

static int DeviceDelete(void *ptr) {
  CUDA_CALL(cudaFree(ptr));
  return 0;
}

static int PinnedNew(void **ptr, size_t size, unsigned int flags) {
  *ptr = GetBuffer(std::this_thread::get_id(), AllocType::Pinned, size);
  return 0;
}

static int PinnedDelete(void *ptr) {
  CUDA_CALL(cudaFreeHost(ptr));
  return 0;
}

nvjpegDevAllocator_t GetDeviceAllocator() {
  nvjpegDevAllocator_t allocator;
  allocator.dev_malloc = &DeviceNew;
  allocator.dev_free = &DeviceDelete;
  return allocator;
}

nvjpegPinnedAllocator_t GetPinnedAllocator() {
  nvjpegPinnedAllocator_t allocator;
  allocator.pinned_malloc = &PinnedNew;
  allocator.pinned_free = &PinnedDelete;
  return allocator;
}

}  // namespace nvjpeg_memory

}  // namespace dali
