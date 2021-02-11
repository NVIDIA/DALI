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
#include "dali/core/error_handling.h"
#include "dali/core/unique_handle.h"

namespace dali {
namespace python {


#if defined(__unix__)
using shm_handle_t = int;

#define POSIX_CHECK_STATUS_EX(status, call_str, message)                               \
  do {                                                                                 \
    if (status == -1) {                                                                \
      std::string errmsg(256, '\0');                                                   \
      int e = errno;                                                                   \
      strerror_r(e, &errmsg[0], errmsg.size());                                        \
      DALI_FAIL(make_string("Call to ", call_str, " failed. ", errmsg, " ", message)); \
    }                                                                                  \
  } while (0)


#define POSIX_CHECK_STATUS(status, call_str) POSIX_CHECK_STATUS_EX(status, call_str, "")

#define POSIX_CALL_EX(code, message)               \
  do {                                             \
    int status = code;                             \
    POSIX_CHECK_STATUS_EX(status, #code, message); \
  } while (0)


#define POSIX_CALL(code) POSIX_CALL_EX(code, "")

#else
#error Platform not supported
#endif


/**
 * @brief Creates or simply stores (if provided) handle used for identifying
 * shared memory chunk. Closes handle in the desctructor if it
 * wasn't closed earlier.
 */
class DLL_PUBLIC ShmHandle : public UniqueHandle<shm_handle_t, ShmHandle> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(shm_handle_t, ShmHandle);

  /**
   * Create new handle (file descriptor on Unix) that can be used for shared memory chunk.
   */
  static ShmHandle CreateHandle();

  static void DestroyHandle(shm_handle_t h);

  static constexpr shm_handle_t null_handle() {
    return -1;
  }
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

struct MappedMemoryChunk {
  uint64_t size;
  uint8_t *ptr;
  bool operator==(const MappedMemoryChunk &other) const {
    return size == other.size && ptr == other.ptr;
  }
};

class DLL_PUBLIC MappedMemoryHandle : public UniqueHandle<MappedMemoryChunk, MappedMemoryHandle> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(MappedMemoryChunk, MappedMemoryHandle);

  MappedMemoryHandle(int fd, uint64_t size);

  void resize(uint64_t size);

  uint8_t *get_raw_ptr();

  static void DestroyHandle(MappedMemoryChunk);

  static constexpr MappedMemoryChunk null_handle() {
    return {0, nullptr};
  }
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
  ShmHandle fd_;
  std::unique_ptr<MapMemWrapper> mem_;
};

}  // namespace python
}  // namespace dali

#endif  // DALI_UTIL_SHARED_MEM_H_
