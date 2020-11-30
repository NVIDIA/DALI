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

#ifndef DALI_CORE_MM_MEMORY_RESOURCE_H_
#define DALI_CORE_MM_MEMORY_RESOURCE_H_

#include <cuda_runtime.h>
#include <cstddef>

namespace dali {

/**
 * @brief Memory Manager
 *
 * This namespace contains classes and function for memory managment.
 * DALI memory manager follows the interface of C++17 polymorphic memory resource for
 * ordinary host allocators and extends them by the CUDA stream support for
 * stream-aware allocators.
 *
 * Some of the memory resources are composable, accepting an upstream memory resource.
 * Such composite resources can be used to quickly build an application-specific memory resource,
 * tailored to specific needs.
 */
namespace mm {

/**
 * @brief This is an interface for generic memory resources.
 */
class memory_resource {
 public:
  memory_resource() = default;
  memory_resource(const memory_resource &) = delete;
  memory_resource &operator=(const memory_resource &) = delete;
  virtual ~memory_resource() = default;

  /**
   * @brief Allocates memory
   * @param bytes       Size, in bytes, of the requested block
   * @param alignment   Alignment, in bytes, of the requested block.
   * @return A pointer to a memory region that satisfies bytes and alignment constraints.
   *         The pointer should, in general, be freed with a call to deallocate.
   *         Some memory resources don't require deallocation.
   */
  void *allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
    return do_allocate(bytes, alignment);
  }

  /**
   * @brief Deallocates memory returned by a prior call to `allocate`
   * @param mem         Pointer to a block of memory returned by `allocate`
   * @param bytes       Size, in bytes, of the block being deallocated
   * @param alignment   Alignment, in bytes, of the block being deallocated.
   */
  void deallocate(void *ptr, size_t bytes, size_t alignment = alignof(std::max_align_t)) {
    do_deallocate(ptr, bytes, alignment);
  }

  /**
   * @brief Checks for equality of the memory resources.
   *
   * Memory resources are considered equal if one can free memory allocated by the other.
   */
  bool operator==(const memory_resource &other) const noexcept {
    return do_is_equal(other);
  }

 private:
  virtual void *do_allocate(size_t bytes, size_t alignment) = 0;
  virtual void do_deallocate(void *ptr, size_t bytes, size_t alignment) = 0;
  virtual bool do_is_equal(const memory_resource &other) const noexcept { return this == &other; }
};

/**
 * @brief This is an interface for memory resources that associate allocations with CUDA streams.
 */
class stream_memory_resource {
 public:
  stream_memory_resource() = default;

  stream_memory_resource(const stream_memory_resource &) = delete;
  stream_memory_resource &operator=(const stream_memory_resource &) = delete;
  virtual ~stream_memory_resource() = default;


  /**
   * @brief Allocates memory for immediate use on stream `stream`.
   * @param stream      CUDA stream on which the memory can be immediately used
   * @param bytes       Size, in bytes, of the requested block
   * @param alignment   Alignment, in bytes, of the requested block.
   * @return A pointer to a memory region that satisfies bytes and alignment constraints and
   *         which can be immediately used on a given CUDA stream.
   *         The pointer should, in general, be freed with a call to deallocate.
   *         Some memory resources don't require deallocation.
   *
   * stream_memory_resource is a special memory resource that is aware of CUDA streams
   * and can process allocation and deallocation in stream order. The memory can be modified
   * immediately on given `stream` but it may be still in use on a different stream.
   * Similarly, memory returned for a stream cannot be used on host until all work scheduled
   * on the stream at the moment of a call to allocate is complete.
   */
  void *allocate(cudaStream_t stream,
                 size_t bytes, size_t alignment = alignof(std::max_align_t)) {
    return do_allocate(stream, bytes, alignment);
  }

  /**
   * @brief Deallocates memory returned by a prior call to `allocate` for immediate use
   *        on stream `stream`.
   * @param stream      CUDA stream on which the memory can be immediately reused; in general,
   *                    it does not need to be the same stream as was passed to `allocate`.
   * @param mem         Pointer to a block of memory returned by `allocate`
   * @param bytes       Size, in bytes, of the block being deallocated
   * @param alignment   Alignment, in bytes, of the block being deallocated.
   *
   * Implementations of stream_memory_resource may place the freed memory in a per-stream
   * pool and subsequent calls to allocate may return this memory block or its parts while it's
   * still in use on device.
   * It is safe use deallocate on memory that is still in use by the
   * indicated stream.
   */
  void deallocate(cudaStream_t stream,
                  void *ptr, size_t bytes, size_t alignment = alignof(std::max_align_t)) {
    do_deallocate(stream, ptr, bytes, alignment);
  }

  /**
   * @brief Checks for equality of the memory resources.
   *
   * Memory resources are considered equal if one can free memory allocated by the other.
   */
  bool operator==(const stream_memory_resource &other) const noexcept {
    return do_is_equal(other);
  }

 private:
  virtual void *do_allocate(cudaStream_t stream, size_t bytes, size_t alignment) = 0;
  virtual void do_deallocate(cudaStream_t stream, void *ptr, size_t bytes, size_t alignment) = 0;
  virtual bool do_is_equal(const stream_memory_resource &other) const noexcept {
      return this == &other;
  }
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MEMORY_RESOURCE_H_
