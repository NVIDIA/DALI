// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SCRATCH_COPY_IMPL_H_
#define DALI_KERNELS_SCRATCH_COPY_IMPL_H_

#include <cuda_runtime.h>
#include <algorithm>
#include <tuple>
#include <memory>
#include "dali/core/traits.h"
#include "dali/kernels/context.h"

namespace dali {
namespace kernels {

namespace detail {

inline void copy_to_buffer(char *buffer, const size_t *offsets) {}

/**
 * @brief Copy contents of collections `{ c, tail... }` to pointers stored in `ptrs`.
 */
template <typename Collection, typename... Collections>
void copy_to_buffer(char *buffer,
                  const size_t *offsets,
                  const Collection &c,
                  const Collections &... tail) {
  using T = std::remove_cv_t<element_t<Collection>>;
  std::copy(dali::begin(c), dali::end(c), reinterpret_cast<T*>(buffer + offsets[0]));
  copy_to_buffer(buffer, offsets+1, tail...);
}

inline void GetCollectionOffsets(size_t base, size_t *offsets) { *offsets = base; }

/**
 * @brief Assuming aligned storage in a single buffer,
 *        calculates start offsets of collections `{ c, tail... }`
 * @param base     - offset of the first element of the first collection `c`
 * @param offsets  - the array to store the offsets
 * @param c        - collection to be stored at (aligned) `base`
 * @param tail     - collections to be stored after `c`
 */
template <typename Collection, typename... Collections>
void GetCollectionOffsets(size_t base, size_t *offsets,
                              const Collection &c,
                              const Collections &...tail) {
  using T = std::remove_cv_t<element_t<Collection>>;
  base = align_up(base, alignof(T));
  *offsets = base;
  base += size(c) * sizeof(T);
  GetCollectionOffsets(base, offsets + 1, tail...);
}

constexpr std::tuple<> GetCollectionPtrs(void *base, const size_t *offsets) { return {}; }

template <typename Collection, typename... Collections>
auto GetCollectionPtrs(void *base, const size_t *offsets,
                       const Collection &c,
                       const Collections &...tail) {
  using T = std::remove_cv_t<element_t<Collection>>;
  return std::tuple_cat(
    std::make_tuple(reinterpret_cast<T*>(static_cast<char *>(base) + offsets[0])),
    GetCollectionPtrs(base, offsets+1, tail...));
}


template <typename T>
T variadic_max(T t) {
  return t;
}

template <typename T0, typename T1>
auto variadic_max(T0 t0, T1 t1) {
  return t1 > t0 ? t1 : t0;
}

template <typename T0, typename... T>
auto variadic_max(T0 t0, T... tail) {
  return variadic_max(t0, variadic_max(tail...));
}

}  // namespace detail


/**
 * @brief Allocates from scratchpad and copies the collections to the allocated buffer.
 */
template <typename... Collections>
std::tuple<std::remove_cv_t<element_t<Collections>>*...>
ToContiguousHostMem(Scratchpad &scratchpad, const Collections &... c) {
  const size_t N = sizeof...(Collections);
  static_assert(
    all_of<std::is_trivially_copyable<std::remove_cv_t<element_t<Collections>>>::value...>::value,
    "ToContiguousHostMem must be used with collections of trivially copyable types");

  std::array<size_t, N + 1> offsets;
  detail::GetCollectionOffsets(0, &offsets[0], c...);
  size_t alignment = detail::variadic_max(alignof(element_t<Collections>)...);
  size_t total_size = std::get<N>(offsets);

  auto *tmp = scratchpad.AllocateHost<char>(total_size, alignment);
  detail::copy_to_buffer(tmp, &offsets[0], c...);

  return detail::GetCollectionPtrs(tmp, &offsets[0], c...);
}

/**
 * @brief Allocates GPU from scratchpad, copies the collections to a
 *        temporary host buffer and then transfers the contents to the GPU in just one
 *        `cudaMemcpyAsync`.
 */
template <typename... Collections>
std::tuple<std::remove_cv_t<element_t<Collections>>*...>
ToContiguousGPUMem(Scratchpad &scratchpad, cudaStream_t stream, const Collections &... c) {
  const size_t N = sizeof...(Collections);
  static_assert(
    all_of<std::is_trivially_copyable<std::remove_cv_t<element_t<Collections>>>::value...>::value,
    "ToContiguousGPUMem must be used with collections of trivially copyable types");

  std::array<size_t, N + 1> offsets;
  detail::GetCollectionOffsets(0, &offsets[0], c...);
  size_t alignment = detail::variadic_max(alignof(element_t<Collections>)...);
  size_t total_size = std::get<N>(offsets);

  char *tmp;
  std::unique_ptr<char[]> heap_buf;
  if (total_size <= 0x2000) {
     tmp = static_cast<char*>(alloca(0x2000));
  } else {
    heap_buf.reset(new char[total_size]);
    tmp = heap_buf.get();
  }

  detail::copy_to_buffer(tmp, &offsets[0], c...);
  void *out_ptr = scratchpad.Alloc<mm::memory_kind::device>(total_size, alignment);

  CUDA_CALL(cudaMemcpyAsync(out_ptr, tmp, total_size, cudaMemcpyHostToDevice, stream));
  return detail::GetCollectionPtrs(out_ptr, &offsets[0], c...);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SCRATCH_COPY_IMPL_H_
