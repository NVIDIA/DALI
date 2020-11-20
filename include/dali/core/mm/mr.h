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

#ifndef DALI_CORE_MM_MR_H_
#define DALI_CORE_MM_MR_H_

#include <cstddef>
#include <cuda_runtime.h>

namespace dali {
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

  stream_memory_resource(const memory_resource &) = delete;
  stream_memory_resource &operator=(const memory_resource &) = delete;
  virtual ~stream_memory_resource() = default;


  /**
   * @brief Allocates memory for immediate use on stream `stream`.
   * @param stream      CUDA stream on which the memory can be immediately reused; in general,
   *                    it does not need to be the same stream as was passed to `allocate`.
   * @param bytes       Size, in bytes, of the requested block
   * @param alignment   Alignment, in bytes, of the requested block.
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

#endif  // DALI_CORE_MM_MR_H_
