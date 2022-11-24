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

#ifndef DALI_IMGCODEC_DECODERS_MEMORY_POOL_H_
#define DALI_IMGCODEC_DECODERS_MEMORY_POOL_H_

#include <atomic>
#include <array>
#include <shared_mutex>
#include <thread>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "dali/core/mm/memory.h"
#include "dali/core/mm/memory_kind.h"

namespace dali {
namespace imgcodec {

struct AllocInfo {
  mm::memory_kind_id kind;
  mm::Deleter deleter;
  size_t size;
  std::thread::id thread_id;
};

struct Deleter {
  inline void operator()(void *p) const;
};

using unique_ptr_t = std::unique_ptr<char, Deleter>;

struct Buffer {
  Buffer(unique_ptr_t unq_ptr, mm::memory_kind_id kind, size_t sz);
  unique_ptr_t ptr;
  mm::memory_kind_id kind;
  size_t size;
};

using MemoryPool = std::array<std::vector<Buffer>, static_cast<size_t>(mm::memory_kind_id::count)>;
using BufferPool = std::map<std::thread::id, MemoryPool>;

struct BufferPoolManager {
  BufferPool buffer_pool_;
  std::shared_timed_mutex buffer_pool_mutex_;

  struct MemoryStats {
    size_t nallocs = 0;
    size_t biggest_alloc = 0;
  };

  std::array<MemoryStats, static_cast<size_t>(mm::memory_kind_id::count)> mem_stats_;
  std::mutex mem_stats_mutex_;
  std::atomic<bool> mem_stats_enabled_ = {true};

  static BufferPoolManager &instance();

  void SetEnableMemStats(bool enabled);

  template <typename MemoryKind>
  void AddMemStats(size_t size);

  void PrintMemStats();

  template <typename MemoryKind>
  unique_ptr_t Allocate(mm::memory_resource<MemoryKind> *mr,
                        std::thread::id thread_id, size_t size);

  template <typename MemoryKind>
  unique_ptr_t Allocate(std::thread::id thread_id, size_t size);

  template <typename MemoryKind>
  void* GetBuffer(std::thread::id thread_id, size_t size);

  int ReturnBufferToPool(void *raw_ptr);

  template <typename MemoryKind>
  void AddBuffer(std::thread::id thread_id, size_t size);

  void DeleteAllBuffers(std::thread::id thread_id);
};

template <>
unique_ptr_t BufferPoolManager::Allocate<mm::memory_kind::device>(std::thread::id thread_id,
                                                                 size_t size);

int ReturnBufferToPool(void *raw_ptr);

int DeviceNew(void **ptr, size_t size);

int PinnedNew(void **ptr, size_t size, unsigned int flags);

int HostNew(void **ptr, size_t size, unsigned int flags);

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_MEMORY_POOL_H_
