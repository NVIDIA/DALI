// Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_CONTEXT_H_
#define DALI_KERNELS_CONTEXT_H_

#include <cuda_runtime_api.h>
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "dali/core/access_order.h"
#include "dali/core/tensor_view.h"
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/memory_kind.h"
#include "dali/core/backend_tags.h"

namespace dali {
namespace kernels {

template <typename ComputeBackend>
struct Context {};

template <>
struct Context<ComputeGPU> {
  cudaStream_t stream = AccessOrder::null_stream();
};

class Scratchpad;

template <typename... Collections>
std::tuple<std::remove_cv_t<element_t<Collections>>*...>
ToContiguousHostMem(Scratchpad &scratchpad, const Collections &... c);

template <typename... Collections>
std::tuple<std::remove_cv_t<element_t<Collections>>*...>
ToContiguousGPUMem(Scratchpad &scratchpad, cudaStream_t stream, const Collections &... c);

/**
 * @brief Interface for kernels to obtain auxiliary working memory
 *
 * A scratchpad object provides memory indefinitely (as long as underlying resources are available)
 * - all memory is deallocated at once, at the point a Scratchpad goes out of scope.
 *
 * This means that a scratchpad object should, in gerneral, be a temporary object.
 */
class Scratchpad {
 public:
  /**
   * @brief Allocates `bytes` bytes of memory of `MemoryKind`, with specified `alignment`.
   */
  template <typename MemoryKind>
  inline void *Alloc(size_t bytes, size_t alignment) {
    return Alloc(mm::kind2id_v<MemoryKind>, bytes, alignment);
  }

  /**
   * @brief Allocates storage for a Tensor of elements `T` and given `shape`
   *        in the memory of type `alloc_type`.
   */
  template <typename MemoryKind, typename T, int dim>
  TensorView<kind2storage_t<MemoryKind>, T, dim> AllocTensor(TensorShape<dim> shape) {
    return { Allocate<MemoryKind, T>(volume(shape)), std::move(shape) };
  }

  /**
   * @brief Allocates storage for a TensorList of elements `T` and given `shape`
   *        in the memory of type `alloc_type`.
   */
  template <typename MemoryKind, typename T, int dim>
  TensorListView<kind2storage_t<MemoryKind>, T, dim>
  AllocTensorList(const std::vector<TensorShape<dim>> &shape) {
    return AllocTensorList<MemoryKind, T, dim>(TensorListShape<dim>(shape));
  }

  /**
   * @brief Allocates storage for a TensorList of elements `T` and given `shape`
   *        in the memory of kind `MemoryKind`.
   */
  template <typename MemoryKind, typename T, int dim>
  TensorListView<kind2storage_t<MemoryKind>, T, dim>
  AllocTensorList(TensorListShape<dim> shape) {
    T *data = Allocate<MemoryKind, T>(shape.num_elements());
    TensorListView<kind2storage_t<MemoryKind>, T, dim> tlv(data, std::move(shape));
    return tlv;
  }

  /**
   * @brief Allocates memory suitable for storing `count` items of type `T` in the
   *        memory of kind `MemoryKind`.
   */
  template <typename MemoryKind, typename T>
  T *Allocate(size_t count, size_t alignment = alignof(T)) {
    return reinterpret_cast<T*>(Alloc<MemoryKind>(count*sizeof(T), alignment));
  }

  /**
   * @brief Allocates memory suitable for storing `count` items of type `T` on GPU
   */
  template <typename T>
  T *AllocateGPU(size_t count, size_t alignment = alignof(T)) {
    return Allocate<mm::memory_kind::device, T>(count, alignment);
  }

  /**
   * @brief Allocates memory suitable for storing `count` items of type `T` in host memory
   */
  template <typename T>
  T *AllocateHost(size_t count, size_t alignment = alignof(T)) {
    return Allocate<mm::memory_kind::host, T>(count, alignment);
  }

  /**
   * @brief Allocates memory suitable for storing `count` items of type `T` in host pinned memory
   */
  template <typename T>
  T *AllocatePinned(size_t count, size_t alignment = alignof(T)) {
    return Allocate<mm::memory_kind::pinned, T>(count, alignment);
  }

  /**
   * @brief Allocates memory suitable for storing `count` items of type `T` in managed memory
   */
  template <typename T>
  T *AllocateManaged(size_t count, size_t alignment = alignof(T)) {
    return Allocate<mm::memory_kind::managed, T>(count, alignment);
  }

  template <typename Collection, typename T = std::remove_const_t<element_t<Collection>>>
  if_array_like<Collection, T*>
  ToGPU(cudaStream_t stream, const Collection &c) {
    auto n = dali::size(c);
    T *ptr = AllocateGPU<T>(n);
    CUDA_CALL(cudaMemcpyAsync(ptr, &c[0], n * sizeof(T), cudaMemcpyHostToDevice, stream));
    return ptr;
  }

  template <typename Collection, typename T = std::remove_const_t<element_t<Collection>>>
  if_iterable<Collection, T*>
  ToHost(const Collection &c) {
    T *ptr = AllocateHost<T>(dali::size(c));
    T *it = ptr;
    for (const auto &x : c)
      new(it++)T(x);
    return ptr;
  }

  template <typename Collection, typename T = std::remove_const_t<element_t<Collection>>>
  if_iterable<Collection, T*>
  ToPinned(const Collection &c) {
    T *ptr = AllocatePinned<T>(dali::size(c));
    T *it = ptr;
    for (const auto &x : c)
      new(it++)T(x);
    return ptr;
  }

  template <typename Collection, typename T = std::remove_const_t<element_t<Collection>>>
  if_iterable<Collection, T*>
  ToManaged(const Collection &c) {
    T *ptr = AllocateManaged<T>(dali::size(c));
    T *it = ptr;
    for (const auto &x : c)
      new(it++)T(x);
    return ptr;
  }

  template <typename... Collections>
  auto ToContiguousHost(const Collections &...collections) {
    return ToContiguousHostMem(*this, collections...);
  }

  template <typename... Collections>
  auto ToContiguousGPU(cudaStream_t stream, const Collections &...collections) {
    return ToContiguousGPUMem(*this, stream, collections...);
  }

  virtual void *Alloc(mm::memory_kind_id kind_id, size_t bytes, size_t alignment) = 0;

 protected:
  ~Scratchpad() = default;
};

using CPUContext = Context<ComputeCPU>;
using GPUContext = Context<ComputeGPU>;

struct KernelContext {
  CPUContext cpu;
  GPUContext gpu;

  /**
   * @brief Caller-provided allocator for temporary data.
   */
  Scratchpad *scratchpad = nullptr;
};

}  // namespace kernels
}  // namespace dali

#include "dali/kernels/scratch_copy_impl.h"

#endif  // DALI_KERNELS_CONTEXT_H_
