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

#ifndef DALI_UTIL_SHARED_MEM_H_
#define DALI_UTIL_SHARED_MEM_H_

#include <stdint.h>
#include <memory>
#include "dali/core/common.h"

namespace dali {
namespace python {

/**
 * @brief Creates or simply stores (if provided) file descriptor used for identifying
 * shared memory chunk. Closes file descriptor in the desctructor if it
 * wasn't closed earlier.
 */
class DLL_PUBLIC ShmFdWrapper {
 public:
  /**
   * @brief Wrap a file descriptor, fd shouldn't be -1.
   */
  explicit ShmFdWrapper(int fd);

  /**
   * Create new file descriptor that can be used for shared memory chunk.
   */
  ShmFdWrapper();

  ~ShmFdWrapper();

  DISABLE_COPY_MOVE_ASSIGN(ShmFdWrapper);

  int get_fd();

  int close();

 private:
  int fd_;
};


/**
 * @brief Map/unmap and resize mapping of shared memory into process address space.
 * Unmaps memory in the destructor if it wasn't unmapped earlier.
 */
class DLL_PUBLIC MapMemWrapper {
 public:
  MapMemWrapper(int fd, uint64_t size);

  ~MapMemWrapper();

  DISABLE_COPY_MOVE_ASSIGN(MapMemWrapper);

  uint8_t *get_raw_ptr();

  void resize(uint64_t size);

  int unmap();

 private:
  uint64_t size_;
  uint8_t *ptr_;
};

/**
 * @brief Allocate or access existing shared memory chunk identified by file descriptor.
 * When created with fd = -1, it allocates new shared memory of size ``size`` and mmaps
 * it into process address space. When fd of existing shared memory chunk is
 * passed the related shared memory chunk will be mmaped into process address space,
 * ``size`` should match the size of underlying shared memory chunk.
 */
class DLL_PUBLIC SharedMem {
 public:
  using b_type = uint8_t;

  DLL_PUBLIC SharedMem(int fd, uint64_t size);

  DLL_PUBLIC ~SharedMem() = default;

  DLL_PUBLIC uint64_t size() const;

  DLL_PUBLIC int fd();

  DLL_PUBLIC b_type *get_raw_ptr();

  /**
   * @brief Resize the current chunk with optional call to ftruncate (to actually change the size)
   * or just remap to the new size.
   */
  DLL_PUBLIC void resize(uint64_t size, bool trunc = false);

  DLL_PUBLIC void close_fd();

  DLL_PUBLIC void close_map();

 private:
  uint64_t size_;
  std::unique_ptr<ShmFdWrapper> fd_;
  std::unique_ptr<MapMemWrapper> mem_;
};

}  // namespace python
}  // namespace dali

#endif  // DALI_UTIL_SHARED_MEM_H_
