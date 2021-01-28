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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <exception>
#include <cstring>

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
ShmFdWrapper::ShmFdWrapper(int fd) : fd_{fd} {}

ShmFdWrapper::ShmFdWrapper() {
  // Abstract away the fact that shm_open requires filename.
  constexpr char dev_shm_path[] = "/dev/shm/";
  constexpr char run_shm_path[] = "/run/shm/";
  constexpr char temp_filename_template[] = "nvidia_dali_XXXXXX";
  constexpr int kShmPathSize = sizeof(dev_shm_path) - 1;
  constexpr int kPathSize = kShmPathSize + sizeof(temp_filename_template);
  char temp_path[kPathSize];
  const char *shm_path = nullptr;
  if (detail::dir_exists(dev_shm_path)) {
    shm_path = dev_shm_path;
  } else if (detail::dir_exists(run_shm_path)) {
    shm_path = run_shm_path;
  } else {
    throw std::runtime_error("shared memory dir not found");
  }
  memcpy(temp_path, shm_path, kShmPathSize);
  memcpy(temp_path + kShmPathSize, temp_filename_template, sizeof(temp_filename_template));
  int temp_fd = mkstemp(temp_path);
  if (temp_fd < 0) {
    throw std::runtime_error("temporary file creation failed");
  }
  if (unlink(temp_path) != 0) {
    throw std::runtime_error("couldn't unlink temporary file");
  }
  const char * temp_filename = temp_path + kShmPathSize;
  fd_ = shm_open(temp_filename, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd_ < 0) {
    throw std::runtime_error("shm_open call failed");
  }
  ::close(temp_fd);
  // The memory chunks will be passed between processes by sending descriptors over the sockets,
  // not through the filenames; this way in case of unexpected process exit we're not leaking any
  // filelike objects in /dev/shm or /tmp. The OS guarantees to keep the memory as long as
  // any process has fds or has mmaped the memory.
  shm_unlink(temp_filename);
}

ShmFdWrapper::~ShmFdWrapper() {
  close();
}

int ShmFdWrapper::get_fd() {
  return fd_;
}

int ShmFdWrapper::close() {
  int ret = 0;
  if (fd_ >= 0) {
    ret = ::close(fd_);
    fd_ = -1;
  }
  return ret;
}

MapMemWrapper::MapMemWrapper(int fd, uint64_t size)
    : size_{size},
      ptr_{static_cast<SharedMem::b_type *>(
          mmap(nullptr, size_, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0))} {
  if (ptr_ == MAP_FAILED) {
    throw std::runtime_error("mmap failed");
  }
}

MapMemWrapper::~MapMemWrapper() {
  unmap();
}

uint8_t *MapMemWrapper::get_raw_ptr() {
  return ptr_;
}

void MapMemWrapper::resize(uint64_t new_size) {
  if (!ptr_) {
    throw std::runtime_error("cannot resize the memory mapping, because no memory is mapped");
  }
  ptr_ = static_cast<SharedMem::b_type *>(mremap(ptr_, size_, new_size, MREMAP_MAYMOVE));
  if (ptr_ == MAP_FAILED) {
    throw std::runtime_error("mremap failed");
  }
  size_ = new_size;
}

int MapMemWrapper::unmap() {
  int ret = 0;
  if (ptr_) {
    ret = munmap(ptr_, size_);
    ptr_ = nullptr;
  }
  return ret;
}

SharedMem::SharedMem(int fd, uint64_t size) : size_{size * sizeof(SharedMem::b_type)} {
  if (fd >= 0) {
    fd_ = std::make_unique<ShmFdWrapper>(fd);
  } else {
    fd_ = std::make_unique<ShmFdWrapper>();
    if (ftruncate(fd_->get_fd(), size_) == -1) {
      throw std::runtime_error("failed to resize shared memory");
    }
  }
  mem_ = std::make_unique<MapMemWrapper>(fd_->get_fd(), size_);
}

uint64_t SharedMem::size() const {
  return size_;
}

int SharedMem::fd() {
  return !fd_ ? -1 : fd_->get_fd();
}

SharedMem::b_type *SharedMem::get_raw_ptr() {
  return !mem_ ? nullptr : mem_->get_raw_ptr();
}

void SharedMem::resize(uint64_t size, bool trunc) {
  size_ = size * sizeof(SharedMem::b_type);
  if (trunc) {
    if (ftruncate(fd_->get_fd(), size_) == -1) {
      throw std::runtime_error("failed to resize shared memory");
    }
  }
  if (mem_) {
    mem_->resize(size_);
  } else {
    if (!fd_) {
      throw std::runtime_error("cannot mmap memory - no file descriptor");
    }
    mem_ = std::make_unique<MapMemWrapper>(fd_->get_fd(), size_);
  }
}

  void SharedMem::close_fd() {
    if (fd_ && fd_->close() != 0) {
      throw std::runtime_error("closing fd failed");
    }
  }

  void SharedMem::close_map() {
    if (mem_ && mem_->unmap() != 0) {
      throw std::runtime_error("unmaping shared memory failed");
    }
  }

  }  // namespace python
}  // namespace dali
