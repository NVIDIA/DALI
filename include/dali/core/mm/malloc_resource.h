// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_MALLOC_RESOURCE_H_
#define DALI_CORE_MM_MALLOC_RESOURCE_H_

#include <stdlib.h>
#include <malloc.h>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/device_guard.h"

namespace dali {
namespace mm {

/**
 * @brief A memory resource that manages host memory with std::aligned_alloc and std::free
 */
class malloc_memory_resource : public host_memory_resource {
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0)
      return nullptr;
    void *ptr = memalign(alignment, bytes + sizeof(int));
    if (!ptr)
      throw std::bad_alloc();
    return ptr;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    return free(ptr);
  }

  bool do_is_equal(const memory_resource &other) const noexcept override {
    return dynamic_cast<const malloc_memory_resource*>(&other) != nullptr;
  }

 public:
  static malloc_memory_resource &instance() {
    static malloc_memory_resource inst;
    return inst;
  }
};

/**
 * @brief A memory resource that directly calls cudaMalloc and cudaFree.
 */
class cuda_malloc_memory_resource : public device_async_resource {
 public:
  /**
   * @param device_id The device identifier. If it's negative, then the resource
   *                  will use the current resource _in every call to allocate_,
   *                  - that's in contrast to checking the active device at construction.
   */
  explicit cuda_malloc_memory_resource(int device_id = -1) : device_id_(device_id) {
  }

  static cuda_malloc_memory_resource &instance() {
    static cuda_malloc_memory_resource inst;
    return inst;
  }

 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0)
      return nullptr;
    DeviceGuard dg(device_id_);
    void *mem = nullptr;
    if (alignment > 256)
      throw dali::CUDABadAlloc();
    CUDA_CALL(cudaMalloc(&mem, bytes | 1));  // |1 to prevent accidental coalescing
    return mem;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    if (ptr) {
      CUDA_DTOR_CALL(cudaFree(ptr));
    }
  }

  void *do_allocate_async(size_t bytes, size_t alignment, stream_view) override {
    return allocate(bytes, alignment);
  }

  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view) override {
    return deallocate(mem, bytes, alignment);
  }

  bool do_is_equal(const memory_resource<memory_kind> &other) const noexcept override {
    return dynamic_cast<const cuda_malloc_memory_resource*>(&other) != nullptr;
  }

  int device_id_ = -1;
};


/**
 * @brief A memory resource that directly calls cudaMallocHost and cudaFreeHost.
 */
class pinned_malloc_memory_resource : public pinned_async_resource {
  const size_t kGuaranteedAlignment = 256;

  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0)
      return nullptr;
    if (alignment <= kGuaranteedAlignment)
      alignment = 1;  // cudaMallocHost guarantees suffcient alignment - avoid overhead

    return detail::aligned_alloc([](size_t size) {
      void *mem = nullptr;
      CUDA_CALL(cudaMallocHost(&mem, size | 1));  // |1 to prevent accidental coalescing
      return mem;
    }, bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    if (ptr) {
      if (alignment <= kGuaranteedAlignment)
        alignment = 1;  // cudaMallocHost guarantees suffcient alignment - avoid overhead
      detail::aligned_dealloc([](void *ptr, size_t) {
        CUDA_DTOR_CALL(cudaFreeHost(ptr));
      }, ptr, bytes, alignment);
    }
  }

  void *do_allocate_async(size_t bytes, size_t alignment, stream_view) override {
    return allocate(bytes, alignment);
  }

  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view) override {
    return deallocate(mem, bytes, alignment);
  }

  bool do_is_equal(const memory_resource<memory_kind> &other) const noexcept override {
    return dynamic_cast<const pinned_malloc_memory_resource*>(&other) != nullptr;
  }

 public:
  static pinned_malloc_memory_resource &instance() {
    static pinned_malloc_memory_resource inst;
    return inst;
  }
};

/**
 * @brief A memory resource that directly calls cudaMallocManaged and cudaFree.
 */
class managed_malloc_memory_resource : public managed_async_resource {
  const size_t kGuaranteedAlignment = 256;

  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0)
      return nullptr;
    if (alignment <= kGuaranteedAlignment)
      alignment = 1;  // cudaMallocManaged guarantees suffcient alignment - avoid overhead

    return detail::aligned_alloc([](size_t size) {
      void *mem = nullptr;
      CUDA_CALL(cudaMallocManaged(&mem, size | 1));  // |1 to prevent accidental coalescing
      return mem;
    }, bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    if (ptr) {
      if (alignment <= kGuaranteedAlignment)
        alignment = 1;  // cudaMallocManaged guarantees suffcient alignment - avoid overhead
      detail::aligned_dealloc([](void *ptr, size_t) {
        CUDA_DTOR_CALL(cudaFree(ptr));
      }, ptr, bytes, alignment);
    }
  }

  void *do_allocate_async(size_t bytes, size_t alignment, stream_view) override {
    return allocate(bytes, alignment);
  }

  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view) override {
    return deallocate(mem, bytes, alignment);
  }

  bool do_is_equal(const memory_resource<memory_kind> &other) const noexcept override {
    return dynamic_cast<const cuda_malloc_memory_resource*>(&other) != nullptr;
  }

 public:
  static managed_malloc_memory_resource &instance() {
    static managed_malloc_memory_resource inst;
    return inst;
  }
};

#if CUDA_VERSION >= 11020

class DLL_PUBLIC cuda_malloc_async_memory_resource
: public mm::async_memory_resource<mm::memory_kind::device> {
 public:
  cuda_malloc_async_memory_resource() : cuda_malloc_async_memory_resource(-1) {}

  explicit cuda_malloc_async_memory_resource(int device_id);

  static bool is_supported(int device_id = -1);

  int device_id() const noexcept { return device_id_; }

 private:
  void *do_allocate(size_t size, size_t alignment) override;

  void do_deallocate(void *ptr, size_t size, size_t alignment) override;

  void *do_allocate_async(size_t size, size_t alignment, stream_view stream) override;

  void do_deallocate_async(void *ptr, size_t size, size_t alignment, stream_view stream) override;

  int device_id_;
  CUDAStreamLease dummy_host_stream_;
};

#endif

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MALLOC_RESOURCE_H_
