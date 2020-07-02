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
#include <cassert>
#include <iostream> // TODO(janton) remove
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

using BufferPool = std::map<std::thread::id, std::map<AllocType, std::vector<Buffer>>>;
std::shared_timed_mutex buffer_pool_mutex_;
BufferPool buffer_pool_;

void* GetBuffer(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  std::shared_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto it = buffer_pool_.find(thread_id);
  assert(it != buffer_pool_.end());
  lock.unlock();

  auto &buffers = it->second[alloc_type];
  if (!buffers.empty()) {
    auto buf = std::move(buffers.back());
    buffers.pop_back();
    if (size <= buf.size)
      return buf.ptr.release();
  }
  // Couldn't find a preallocated buffer, proceed to allocate
  return kernels::memory::Allocate(alloc_type, size);
}

void AddBuffer(std::thread::id thread_id, AllocType alloc_type, size_t size) {
  std::unique_lock<std::shared_timed_mutex> lock(buffer_pool_mutex_);
  auto& thread_buffer_pool = buffer_pool_[thread_id];
  lock.unlock();

  auto &buffers = thread_buffer_pool[alloc_type];
  buffers.emplace_back(kernels::memory::alloc_unique<char>(alloc_type, size), alloc_type, size);
}

void DeleteAllBuffers(std::thread::id thread_id) {
  buffer_pool_.erase(thread_id);
}

static int DeviceNew(void **ptr, size_t size) {
  std::thread::id this_id = std::this_thread::get_id();
  std::cout << "Requesting device memory for Thread " << this_id << ": " << size << " bytes, ptr = ";
  *ptr = GetBuffer(this_id, AllocType::GPU, size);
  std::cout << *ptr << "\n";
  return 0;
}

static int DeviceDelete(void *ptr) {
  std::thread::id this_id = std::this_thread::get_id();
  std::cout << "Thread " << this_id << " deleting device memory: ptr = " << ptr << "\n";
  CUDA_CALL(cudaFree(ptr));
  return 0;
}

static int PinnedNew(void **ptr, size_t size, unsigned int flags) {
  std::thread::id this_id = std::this_thread::get_id();
  std::cout << "Requesting pinned memory for Thread " << this_id << ": " << size << " bytes, ptr = ";
  *ptr = GetBuffer(this_id, AllocType::Pinned, size);
  std::cout << *ptr << "\n";
  return 0;
}

static int PinnedDelete(void *ptr) {
  std::thread::id this_id = std::this_thread::get_id();
  std::cout << "Thread " << this_id << " deleting pinned memory: ptr = " << ptr << "\n";
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
