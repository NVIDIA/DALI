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

#ifndef DALI_KERNELS_ALLOC_H_
#define DALI_KERNELS_ALLOC_H_

#include <functional>
#include <memory>
#include <type_traits>
#include "dali/kernels/alloc_type.h"
#include "dali/core/api_helper.h"

namespace dali {
namespace kernels {
namespace memory {

DLL_PUBLIC void *Allocate(AllocType type, size_t size) noexcept;
DLL_PUBLIC void Deallocate(AllocType type, void *mem, int device) noexcept;

struct Deleter {
  int device;
  AllocType alloc_type;
  inline void operator()(void *p) noexcept { Deallocate(alloc_type, p, device); }
};
DLL_PUBLIC Deleter GetDeleter(AllocType type) noexcept;

template <typename T>
std::shared_ptr<T> alloc_shared(AllocType type, size_t count) {
  static_assert(std::is_pod<T>::value, "Only POD types are supported");
  void *mem = Allocate(type, count*sizeof(T));
  if (!mem)
    throw std::bad_alloc();

  // From cppreference: if additional storage cannot be allocated
  // deleter will be called on the pointer passed to shared_ptr constructor.
  return { static_cast<T*>(mem), GetDeleter(type) };
}

template <typename T>
using KernelUniquePtr = std::unique_ptr<T, Deleter>;

template <typename T>
KernelUniquePtr<T> alloc_unique(AllocType type, size_t count) {
  static_assert(std::is_pod<T>::value, "Only POD types are supported");
  void *mem = Allocate(type, count*sizeof(T));
  if (!mem)
    throw std::bad_alloc();
  return { reinterpret_cast<T*>(mem), GetDeleter(type) };
}

}  // namespace memory
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ALLOC_H_
