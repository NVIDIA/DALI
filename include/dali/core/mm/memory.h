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

#ifndef DALI_CORE_MM_MEMORY_H_
#define DALI_CORE_MM_MEMORY_H_

#include <memory>
#include "dali/core/mm/default_resources.h"

namespace dali {
namespace mm {

static const cudaStream_t host_sync = ((cudaStream_t)42);  // NOLINT

template <memory_kind kind, typename Context>
struct mr_deallocator {
  memory_resource<kind, Context> *mr;
  size_t size, alignment;
  void operator()(void *memory) noexcept {
    mr->deallocate(memory, size, alignment);
  }
};

template <memory_kind kind>
struct mr_stream_deallocator {
  async_memory_resource<kind> *mr;
  size_t size, alignment;
  cudaStream_t release_on_stream;
  void operator()(void *memory) noexcept {
    if (release_on_stream == host_sync)
      mr->deallocate(memory, size, alignment);
    else
      mr->deallocate_async(memory, size, alignment, release_on_stream);
  }
};

template <typename T, memory_kind kind, typename Context>
using mr_unique_ptr = std::unique_ptr<T, mr_deallocator<kind, Context>>;

template <typename T, memory_kind kind>
using mr_stream_unique_ptr = std::unique_ptr<T, mr_stream_deallocator<kind>>;

template <typename T, memory_kind kind>
void set_dealloc_stream(mr_stream_unique_ptr<T, kind> &ptr, cudaStream_t stream) {
  ptr.get_deleter().release_on_stream = stream;
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr`.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr        Memory resources to allocate the memory from.
 * @param count     Number of objects for which the storage should suffice.
 * @tparam T        Type of the object for which the storage is allocated.
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <typename T, memory_kind kind, typename Context>
mr_unique_ptr<T, kind, Context> alloc_raw(memory_resource<kind, Context> *mr, size_t count) {
  size_t bytes = sizeof(T) * count;
  size_t alignment = alignof(T);
  void *mem = mr->allocate(bytes, alignment);
  return mr_unique_ptr<T, kind, Context>(static_cast<T*>(mem),
            mr_deallocator<kind, Context>{mr, bytes, alignment});
}


/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from default memory resource.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param count     Number of objects for which the storage should suffice.
 * @tparam T        Type of the object for which the storage is allocated.
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <typename T, memory_kind kind>
auto alloc_raw(size_t count) {
  return alloc_raw<T>(GetDefaultResource<kind>(), count);
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *        with stream-ordered semantics
 *
 * This function allocates memory with stream-ordered semantics. The `alloc_stream` denotes
 * the stream on which the memory can be safely used without additional synchronization.
 * Use the value `host_sync` if the memory needs to be accessible on all streams or on host as
 * soon as the function returns.
 * `dealloc_stream` denotest the stream for deallocation. Stream-ordered semantics guarantee,
 * that if there's still some work pending on `dealloc_stream`, it will finish before the memory
 * returned by this function is freed. Use `host_sync` for host-synchronous execution.
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr` with stream semantics.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr              Memory resources to allocate the memory from
 * @param count           Number of objects for which the storage should suffice
 * @param alloc_stream    The CUDA stream on which the memory is immediately usable
 * @param dealloc_stream  The CUDA stream which is guaranteed to finish all work scheduled
 *                        before the deallocation of the memory.
 * @tparam T              Type of the object for which the storage is allocated.
 * @tparam kind           The kind of requested memory.
 */
template <typename T, memory_kind kind>
mr_stream_unique_ptr<T, kind> alloc_raw_async(async_memory_resource<kind> *mr,
                                              size_t count,
                                              cudaStream_t alloc_stream,
                                              cudaStream_t dealloc_stream) {
  size_t bytes = sizeof(T) * count;
  size_t alignment = alignof(T);
  void *mem = alloc_stream == host_sync
    ? mr->allocate(bytes, alignment)
    : mr->allocate_async(bytes, alignment, alloc_stream);
  return mr_stream_unique_ptr<T, kind>(static_cast<T*>(mem),
      mr_stream_deallocator<kind>{mr, bytes, alignment, dealloc_stream});
}

template <typename T, memory_kind kind>
auto alloc_raw_async(size_t count,
                     cudaStream_t alloc_stream,
                     cudaStream_t dealloc_stream) {
  return alloc_raw_async<T, kind>(GetDefaultResource<kind>(), count, alloc_stream, dealloc_stream);
}

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MEMORY_H_
