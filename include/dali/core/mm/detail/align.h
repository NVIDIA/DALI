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

#ifndef DALI_CORE_MM_DETAIL_ALIGN_H_
#define DALI_CORE_MM_DETAIL_ALIGN_H_

#include <cassert>
#include <utility>
#include "dali/core/util.h"

namespace dali {
namespace mm {
namespace detail {

DALI_HOST_DEV
constexpr uintptr_t ptr2u(void *p) noexcept {
  // reinterpret_cast is not a constant expression - use a series of static casts instead
  return static_cast<char*>(p) - static_cast<char*>(nullptr);
}

template <typename T = void>
DALI_HOST_DEV
constexpr T *u2ptr(uintptr_t i) noexcept {
  // reinterpret_cast is not a constant expression - use a series of static casts instead
  return static_cast<T*>(static_cast<void *>(static_cast<char*>(nullptr) + i));
}

template <typename T, typename A>
DALI_HOST_DEV
constexpr T *align_ptr(T *ptr, A alignment) noexcept {
  return u2ptr<T>(align_up(ptr2u(ptr), alignment));
}

template <typename T, typename A>
DALI_HOST_DEV
constexpr T *align_ptr_down(T *ptr, A alignment) noexcept {
  // Cast to proper unsigned type first, then clear LSB.
  // Negation clears least significant bits, producing the desired mask.
  return u2ptr<T>(ptr2u(ptr) & -static_cast<uintptr_t>(alignment));
}


DALI_HOST_DEV
constexpr bool is_aligned(void *ptr, size_t alignment) noexcept {
  return (ptr2u(ptr) & (alignment-1)) == 0;
}

DALI_NO_EXEC_CHECK
template <typename OffsetType, typename AllocFn>
DALI_HOST_DEV
void *aligned_alloc_impl(AllocFn base_alloc, size_t size, size_t alignment) {
  void *ptr = base_alloc(size + alignment + sizeof(OffsetType) - 1);
  OffsetType *data = static_cast<OffsetType *>(ptr);
  OffsetType *aligned = align_ptr(data + 1, alignment);
  OffsetType offset = reinterpret_cast<ptrdiff_t>(aligned) - reinterpret_cast<ptrdiff_t>(data);
  aligned[-1] = offset;
  return aligned;
}

DALI_NO_EXEC_CHECK
template <typename OffsetType, typename DeallocFn>
DALI_HOST_DEV
void aligned_dealloc_impl(DeallocFn base_dealloc, void *mem, size_t size, size_t alignment) {
  OffsetType *ptr = static_cast<OffsetType *>(mem);
  ptrdiff_t offset = ptr[-1];
  if (sizeof(OffsetType) < sizeof(ptrdiff_t) && offset == 0)
    offset = static_cast<OffsetType>(-1) + 1;  // treat 0 as 2^n
  base_dealloc(static_cast<char*>(mem) - offset, size + alignment + sizeof(OffsetType) - 1);
}

DALI_NO_EXEC_CHECK
template <typename AllocFn>
DALI_HOST_DEV
inline void *aligned_alloc(AllocFn base_alloc, size_t size, size_t alignment) {
  if (alignment < 2)
    return base_alloc(size);
  assert(is_pow2(alignment));
  if (alignment > 256)
    return aligned_alloc_impl<ptrdiff_t>(std::forward<AllocFn>(base_alloc), size, alignment);
  else
    return aligned_alloc_impl<uint8_t>(std::forward<AllocFn>(base_alloc), size, alignment);
}

DALI_NO_EXEC_CHECK
template <typename DeallocFn>
DALI_HOST_DEV
inline void aligned_dealloc(DeallocFn base_dealloc, void *mem, size_t size, size_t alignment) {
  if (alignment < 2)
    base_dealloc(mem, size);
  else if (alignment > 256)
    aligned_dealloc_impl<ptrdiff_t>(std::forward<DeallocFn>(base_dealloc), mem, size, alignment);
  else
    aligned_dealloc_impl<uint8_t>(std::forward<DeallocFn>(base_dealloc), mem, size, alignment);
}

}  // namespace detail
}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_ALIGN_H_
