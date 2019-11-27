// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include "dali/kernels/alloc.h"
#include "dali/core/static_switch.h"
#include "dali/core/device_guard.h"

namespace dali {
namespace kernels {
namespace memory {

template <AllocType>
struct Allocator;

template <>
struct Allocator<AllocType::Host> {
  static void Deallocate(void *ptr, int device) noexcept {
    (void)device;
    free(ptr);
  }

  static void *Allocate(size_t bytes) noexcept { return malloc(bytes); }
};

template <>
struct Allocator<AllocType::Pinned> {
  static void Deallocate(void *ptr, int device) noexcept {
    try {
      DeviceGuard guard(device);
    } catch (...) {
      std::terminate();
    }
    cudaFreeHost(ptr);
  }

  static void *Allocate(size_t bytes) noexcept {
    void *ptr = nullptr;
    cudaMallocHost(&ptr, bytes);
    return ptr;
  }
};

template <>
struct Allocator<AllocType::GPU> {
  static void Deallocate(void *ptr, int device) noexcept {
    try {
      DeviceGuard guard(device);
    } catch (...) {
      std::terminate();
    }
    cudaFree(ptr);
  }

  static void *Allocate(size_t bytes) noexcept {
    void *ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
  }
};


template <>
struct Allocator<AllocType::Unified> {
  static void Deallocate(void *ptr, int device) noexcept {
    try {
      DeviceGuard guard(device);
    } catch (...) {
      std::terminate();
    }
    cudaFree(ptr);
  }

  static void *Allocate(size_t bytes) noexcept {
    void *ptr = nullptr;
    cudaMallocManaged(&ptr, bytes);
    return ptr;
  }
};

void *Allocate(AllocType type, size_t size) noexcept {
  VALUE_SWITCH(type, type_label,
    (AllocType::Host, AllocType::Pinned, AllocType::GPU, AllocType::Unified),
    (return Allocator<type_label>::Allocate(size)),
    (assert(!"Invalid allocation type requested");
    return nullptr;)
  );  // NOLINT
}

void Deallocate(AllocType type, void *mem, int device) noexcept {
  VALUE_SWITCH(type, type_label,
    (AllocType::Host, AllocType::Pinned, AllocType::GPU, AllocType::Unified),
    (return Allocator<type_label>::Deallocate(mem, device)),
    (assert(!"Invalid allocation type requested");)
  );  // NOLINT
}

Deleter GetDeleter(AllocType type) noexcept {
  Deleter del;
  del.alloc_type = type;
  del.device = 0;
  if (type != AllocType::Host)
    cudaGetDevice(&del.device);
  return del;
}

}  // namespace memory
}  // namespace kernels
}  // namespace dali
