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

#ifndef DALI_CORE_OS_SHARED_MEM_H_
#define DALI_CORE_OS_SHARED_MEM_H_

#include <stdint.h>
#include <memory>
#include <string>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/unique_handle.h"

namespace dali {

#if defined(__unix__)
using shm_handle_t = int;
using fd_handle_t = int;

void handle_strerror(int errnum, char *buf, size_t buflen) {
  #if (_POSIX_C_SOURCE >= 200112L) && !_GNU_SOURCE
    DALI_ENFORCE(strerror_r(errnum, buf, buflen) == 0, "Call to strerror_r failed.");
  #else
    char *ptr = strerror_r(errnum, buf, buflen);
    if (ptr != buf)
      memcpy(buf, ptr, buflen);
  #endif
}

#define POSIX_CHECK_STATUS_EX(status, call_str, message)                               \
  do {                                                                                 \
    if (status == -1) {                                                                \
      std::string errmsg(256, '\0');                                                   \
      int e = errno;                                                                   \
      handle_strerror(e, &errmsg[0], errmsg.size());                                   \
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

class DLL_PUBLIC FileHandle : public UniqueHandle<fd_handle_t, FileHandle> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(fd_handle_t, FileHandle);

  static void DestroyHandle(fd_handle_t h);

  static constexpr fd_handle_t null_handle() {
    return -1;
  }
};


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


struct MappedMemoryChunk {
  uint64_t size;
  uint8_t *ptr;
  bool operator==(const MappedMemoryChunk &other) const {
    return size == other.size && ptr == other.ptr;
  }
};

/**
 * @brief Map/unmap and resize mapping of shared memory into process address space.
 * Unmaps memory in the destructor if it wasn't unmapped earlier.
 */
class DLL_PUBLIC MemoryMapping : public UniqueHandle<MappedMemoryChunk, MemoryMapping> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(MappedMemoryChunk, MemoryMapping);

  MemoryMapping(shm_handle_t handle, uint64_t size);

  void resize(uint64_t size);

  uint8_t *get_raw_ptr();

  static void DestroyHandle(MappedMemoryChunk);

  static constexpr MappedMemoryChunk null_handle() {
    return {0, nullptr};
  }
};


/**
 * @brief Allocate or access existing shared memory chunk identified by handle (file descriptor).
 * When created with handle = -1, it allocates new shared memory of size ``size`` and mmaps
 * it into process address space. When handle of existing shared memory chunk is
 * passed the related shared memory chunk will be mmaped into process address space,
 * ``size`` should match the size of underlying shared memory chunk.
 */
class DLL_PUBLIC SharedMem {
 public:
  DLL_PUBLIC SharedMem(shm_handle_t handle, uint64_t size);

  DLL_PUBLIC ~SharedMem() = default;

  DLL_PUBLIC uint64_t size() const;

  DLL_PUBLIC shm_handle_t handle();

  DLL_PUBLIC uint8_t *get_raw_ptr();

  /**
   * @brief Resize the current chunk with optional call to ftruncate (to actually change the size)
   * or just remap to the new size.
   */
  DLL_PUBLIC void resize(uint64_t size, bool trunc = false);

  DLL_PUBLIC void close();

 private:
  uint64_t size_;
  ShmHandle shm_handle_;
  MemoryMapping memory_mapping_;
};

}  // namespace dali

#endif  // DALI_CORE_OS_SHARED_MEM_H_
