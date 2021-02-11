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

#include <errno.h>
#include <fcntl.h> /* For O_* constants */
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <exception>

#include "dali/core/format.h"
#include "dali/util/shared_mem.h"

namespace dali {

namespace detail {

bool dir_exists(const char * str) {
  struct stat sb;
  if (stat(str, &sb) == -1) {
    return false;
  }
  return S_ISDIR(sb.st_mode);
}

}  // namespace detail

namespace python {

ShmHandle ShmHandle::CreateHandle() {
  // Abstract away the fact that shm_open requires filename.
  constexpr char dev_shm_path[] = "/dev/shm/";
  constexpr char run_shm_path[] = "/run/shm/";
  constexpr char temp_filename_template[] = "nvidia_dali_XXXXXX";
  constexpr int kDevShmPathSize = sizeof(dev_shm_path) - 1;
  constexpr int kRunShmPathSize = sizeof(run_shm_path) - 1;
  constexpr int kPathSizeMax =
      std::max(kDevShmPathSize, kRunShmPathSize) + sizeof(temp_filename_template);
  char temp_path[kPathSizeMax];
  const char *shm_path = nullptr;
  size_t shm_size = 0;
  if (detail::dir_exists(dev_shm_path)) {
    shm_path = dev_shm_path;
    shm_size = kDevShmPathSize;
  } else if (detail::dir_exists(run_shm_path)) {
    shm_path = run_shm_path;
    shm_size = kRunShmPathSize;
  } else {
    throw std::runtime_error(make_string("Shared memory dir not found, looked for: {", dev_shm_path,
                                         ", ", run_shm_path, "}."));
  }
  memcpy(temp_path, shm_path, shm_size);
  memcpy(temp_path + shm_size, temp_filename_template, sizeof(temp_filename_template));
  int temp_fd = mkstemp(temp_path);  // posix
  POSIX_CHECK_STATUS_EX(temp_fd, "mkstemp", "Temporary file creation failed.");
  POSIX_CALL_EX(unlink(temp_path), "Couldn't unlink temporary file.");
  const char *temp_filename = temp_path + shm_size;
  shm_handle_t fd = shm_open(temp_filename, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  POSIX_CHECK_STATUS(temp_fd, "shm_open");
  ::close(temp_fd);
  // The memory chunks will be passed between processes by sending descriptors over the sockets,
  // not through the filenames; this way in case of unexpected process exit we're not leaking any
  // filelike objects in /dev/shm or /tmp. The OS guarantees to keep the memory as long as
  // any process has fds or has mmaped the memory.
  shm_unlink(temp_filename);
  return ShmHandle(fd);
}

void ShmHandle::DestroyHandle(shm_handle_t h) {
  if (h >= 0) {
    POSIX_CALL(::close(h));
  }
}

MemoryMapping::MemoryMapping(shm_handle_t handle, uint64_t size)
    : UniqueHandle<MappedMemoryChunk, MemoryMapping>(
          {size, static_cast<uint8_t *>(
                     mmap(nullptr, size, PROT_WRITE | PROT_READ, MAP_SHARED, handle, 0))}) {
  if (handle_.ptr == MAP_FAILED) {
    POSIX_CHECK_STATUS(-1, "mmap");
  }
}

void MemoryMapping::DestroyHandle(MappedMemoryChunk handle) {
  if (handle.ptr) {
    POSIX_CALL(munmap(handle.ptr, handle.size));
  }
}

uint8_t *MemoryMapping::get_raw_ptr() {
  return handle_.ptr;
}


void MemoryMapping::resize(uint64_t new_size) {
  if (!handle_.ptr) {
    throw std::runtime_error("cannot resize the memory mapping, because no memory is mapped");
  }
  handle_.ptr = static_cast<uint8_t *>(mremap(handle_.ptr, handle_.size, new_size, MREMAP_MAYMOVE));
  if (handle_.ptr == MAP_FAILED) {
    POSIX_CHECK_STATUS(-1, "mremap");
  }
  handle_.size = new_size;
}

SharedMem::SharedMem(int fd, uint64_t size) : size_{size * sizeof(SharedMem::b_type)} {
  if (fd >= 0) {
    fd_ = ShmHandle(fd);
  } else {
    fd_ = ShmHandle::CreateHandle();
    if (ftruncate(fd_, size_) == -1) {
      constexpr int buf_len = 250;
      char buf[buf_len];
      strerror_r(errno, buf, buf_len);
      throw std::runtime_error(make_string("failed to resize shared memory", buf));
    }
  }
  mem_ = MemoryMapping(fd_, size_);
}

uint64_t SharedMem::size() const {
  return size_;
}

int SharedMem::fd() {
  return !fd_ ? -1 : fd_;
}

SharedMem::b_type *SharedMem::get_raw_ptr() {
  return !mem_ ? nullptr : mem_.get_raw_ptr();
}

void SharedMem::resize(uint64_t size, bool trunc) {
  size_ = size * sizeof(SharedMem::b_type);
  if (trunc) {
    if (ftruncate(fd_, size_) == -1) {
      throw std::runtime_error("failed to resize shared memory");
    }
  }
  if (mem_) {
    mem_.resize(size_);
  } else {
    if (!fd_) {
      throw std::runtime_error("cannot mmap memory - no file descriptor");
    }
    mem_ = MemoryMapping(fd_, size_);
  }
}

void SharedMem::close_fd() {
  if (fd_) {
    fd_.reset();
    // throw std::runtime_error("closing fd failed");
  }
}

  void SharedMem::close_map() {
    if (mem_) {
      mem_.reset();
      // throw std::runtime_error("unmaping shared memory failed");
    }
  }

  }  // namespace python
}  // namespace dali
