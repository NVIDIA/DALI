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

#ifndef DALI_CORE_MM_MALLOC_RESOURCE_H_
#define DALI_CORE_MM_MALLOC_RESOURCE_H_

#include <stdlib.h>
#include <malloc.h>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/cuda_error.h"

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
class cuda_malloc_memory_resource : public memory_resource<memory_kind::device> {
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0)
      return nullptr;
    void *mem = nullptr;
    if (alignment > 256)
      throw dali::CUDABadAlloc();
    CUDA_CALL(cudaMalloc(&mem, bytes + sizeof(int)));
    return mem;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    if (ptr) {
      CUDA_DTOR_CALL(cudaFree(ptr));
    }
  }

  bool do_is_equal(const memory_resource &other) const noexcept override {
    return dynamic_cast<const cuda_malloc_memory_resource*>(&other) != nullptr;
  }

 public:
  static cuda_malloc_memory_resource &instance() {
    static cuda_malloc_memory_resource inst;
    return inst;
  }
};


}  // namespace mm
}  // namespace dali


#endif  // DALI_CORE_MM_MALLOC_RESOURCE_H_
