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

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <utility>
#include <unordered_map>
#include "dali/imgcodec/decoders/nvjpeg/memory_pool.h"
#include "dali/core/cuda_error.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {

namespace imgcodec {

namespace nvjpeg_memory {

std::unordered_map<void*, AllocInfo> alloc_info_;
std::shared_timed_mutex alloc_info_mutex_;

void Deleter::operator()(void *p) const {
  AllocInfo ai;
  {
      std::lock_guard<std::shared_timed_mutex> lock(alloc_info_mutex_);
      auto it = alloc_info_.find(p);
      assert(it != alloc_info_.end());
      ai = std::move(it->second);
      alloc_info_.erase(it);
  }
  ai.deleter(p);
}

Buffer::Buffer(unique_ptr_t unq_ptr, mm::memory_kind_id kind, size_t sz)
: ptr(std::move(unq_ptr)), kind(kind), size(sz) {}

NVJpegMem& NVJpegMem::instance() {
  // ensure proper destruction order
  (void)mm::GetDefaultResource<mm::memory_kind::host>();
  (void)mm::GetDefaultResource<mm::memory_kind::pinned>();
  (void)mm::cuda_malloc_memory_resource::instance();
  static NVJpegMem mem;
  return mem;
}

void NVJpegMem::SetEnableMemStats(bool enabled) {
  mem_stats_enabled_ = enabled;
}

template <typename MemoryKind>
void NVJpegMem::AddMemStats(size_t size) {
  if (mem_stats_enabled_) {
    std::lock_guard<std::mutex> lock(mem_stats_mutex_);
    auto &stats = mem_stats_[static_cast<size_t>(mm::kind2id_v<MemoryKind>)];
    stats.nallocs++;
    if (size > stats.biggest_alloc)
      stats.biggest_alloc = size;
  }
}

void NVJpegMem::PrintMemStats() {
  if (mem_stats_enabled_) {
    std::lock_guard<std::mutex> lock(mem_stats_mutex_);

    const char* log_filename = std::getenv("DALI_LOG_FILE");
    std::ofstream log_file;
    if (log_filename) log_file.open(log_filename);
    std::ostream& out = log_filename ? log_file : std::cout;
    out << std::dec;  // Don't want numbers printed as hex
    out << "#################### NVJPEG STATS ####################" << std::endl;
    auto &device_mem_stats = mem_stats_[static_cast<size_t>(mm::memory_kind_id::device)];
    out << "Device memory: " << device_mem_stats.nallocs
        << " allocations, largest = " << device_mem_stats.biggest_alloc << " bytes\n";
    auto &pinned_mem_stats = mem_stats_[static_cast<size_t>(mm::memory_kind_id::pinned)];
    out << "Host (pinned) memory: " << pinned_mem_stats.nallocs
        << " allocations, largest = " << pinned_mem_stats.biggest_alloc << " bytes\n";
    auto &host_mem_stats = mem_stats_[static_cast<size_t>(mm::memory_kind_id::host)];
    out << "Host (regular) memory: " << host_mem_stats.nallocs
        << " allocations, largest = " << host_mem_stats.biggest_alloc << " bytes\n";
    out << "################## END NVJPEG STATS ##################" << std::endl;
  }
}

template <typename MemoryKind>
unique_ptr_t NVJpegMem::Allocate(mm::memory_resource<MemoryKind> *mr,
                      std::thread::id thread_id, size_t size) {
  auto ptr = mm::alloc_raw_unique<char>(mr, size);
  std::unique_lock<std::shared_timed_mutex> lock(alloc_info_mutex_);
  AllocInfo &ai = alloc_info_[ptr.get()];  // create entry first (before moving out the deleter)
  mm::memory_kind_id kind = mm::kind2id_v<MemoryKind>;
  ai = { kind, std::move(ptr.get_deleter()), size, thread_id};
  return {ptr.release(), {}};
}

template <typename MemoryKind>
unique_ptr_t NVJpegMem::Allocate(std::thread::id thread_id, size_t size) {
  return Allocate(mm::GetDefaultResource<MemoryKind>(), thread_id, size);
}

template <typename MemoryKind>
void* NVJpegMem::GetBuffer(std::thread::id thread_id, size_t size) {
  std::shared_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  mm::memory_kind_id kind = mm::kind2id_v<MemoryKind>;
  auto it = buffer_pool_.find(thread_id);
  auto end_it =  buffer_pool_.end();
  lock.unlock();
  // only if pool exits for given thread_id search it, otherwise just allocate
  if (it != end_it) {
    auto &buffers = it->second[static_cast<size_t>(kind)];
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
  AddMemStats<MemoryKind>(size);
  return Allocate<MemoryKind>(thread_id, size).release();
}

template void* NVJpegMem::GetBuffer<mm::memory_kind::device>(std::thread::id thread_id,
                                                             size_t size);
template void* NVJpegMem::GetBuffer<mm::memory_kind::pinned>(std::thread::id thread_id,
                                                             size_t size);
template void* NVJpegMem::GetBuffer<mm::memory_kind::host>(std::thread::id thread_id,
                                                           size_t size);

int NVJpegMem::ReturnBufferToPool(void *raw_ptr) {
  std::shared_lock<std::shared_timed_mutex> info_lock(alloc_info_mutex_);
  auto info_it = alloc_info_.find(raw_ptr);
  assert(info_it != alloc_info_.end());
  auto info = info_it->second;
  info_lock.unlock();
  std::unique_ptr<char, Deleter> ptr(reinterpret_cast<char*>(raw_ptr), {});
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
  auto &buffers = (*pool)[static_cast<size_t>(info.kind)];
  buffers.emplace_back(std::move(ptr), info.kind, info.size);
  return 0;
  }

template <typename MemoryKind>
void NVJpegMem::AddBuffer(std::thread::id thread_id, size_t size) {
  std::unique_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto kind = mm::kind2id_v<MemoryKind>;
  auto& thread_buffer_pool = buffer_pool_[thread_id];
  lock.unlock();

  auto &buffers = thread_buffer_pool[static_cast<size_t>(kind)];
  buffers.emplace_back(Allocate<MemoryKind>(thread_id, size), kind, size);
  AddMemStats<MemoryKind>(size);
}

template void NVJpegMem::AddBuffer<mm::memory_kind::device>(std::thread::id thread_id, size_t size);
template void NVJpegMem::AddBuffer<mm::memory_kind::pinned>(std::thread::id thread_id, size_t size);
template void NVJpegMem::AddBuffer<mm::memory_kind::host>(std::thread::id thread_id, size_t size);

void NVJpegMem::DeleteAllBuffers(std::thread::id thread_id) {
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

template <>
unique_ptr_t NVJpegMem::Allocate<mm::memory_kind::device>(std::thread::id thread_id, size_t size) {
  // use plain cudaMalloc
  return Allocate(&mm::cuda_malloc_memory_resource::instance(), thread_id, size);
}

template <typename MemoryKind>
void *GetBuffer(std::thread::id thread_id, size_t size) {
  return NVJpegMem::instance().GetBuffer<MemoryKind>(thread_id, size);
}

template void *GetBuffer<mm::memory_kind::device>(std::thread::id thread_id, size_t size);
template void *GetBuffer<mm::memory_kind::pinned>(std::thread::id thread_id, size_t size);
template void *GetBuffer<mm::memory_kind::host>(std::thread::id thread_id, size_t size);

int ReturnBufferToPool(void *raw_ptr) {
  return NVJpegMem::instance().ReturnBufferToPool(raw_ptr);
}

int DeviceNew(void **ptr, size_t size) {
  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }
  // this function should not throw, but return a proper result
  try {
    *ptr = GetBuffer<mm::memory_kind::device>(std::this_thread::get_id(), size);
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

int PinnedNew(void **ptr, size_t size, unsigned int flags) {
  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }
  // this function should not throw, but return a proper result
  try {
    *ptr = GetBuffer<mm::memory_kind::pinned>(std::this_thread::get_id(), size);
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

int HostNew(void **ptr, size_t size, unsigned int flags) {
  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }
  // this function should not throw, but return a proper result
  try {
    *ptr = RestrictPinnedMemUsage()
        ? GetBuffer<mm::memory_kind::host>(std::this_thread::get_id(), size)
        : GetBuffer<mm::memory_kind::pinned>(std::this_thread::get_id(), size);
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

}  // namespace nvjpeg_memory

}  // namespace imgcodec

}  // namespace dali
