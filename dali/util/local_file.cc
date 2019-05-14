// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#if !defined(__AARCH64_GNU__)
#include <linux/sysctl.h>
#include <sys/syscall.h>
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <algorithm>
#include <tuple>

#include "dali/util/local_file.h"
#include "dali/core/error_handling.h"

static int _sysctl(struct __sysctl_args *args);

static int get_max_vm_cnt() {
  int vm_cnt = 1;
#if !defined(__AARCH64_GNU__)
  size_t vm_cnt_sz;
  int name[] = { CTL_VM, VM_MAX_MAP_COUNT };
  struct __sysctl_args args = {0, };

  args.name = name;
  args.nlen = sizeof(name)/sizeof(name[0]);
  args.oldval = &vm_cnt;
  args.oldlenp = &vm_cnt_sz;

  vm_cnt_sz = sizeof(vm_cnt);
  if (syscall(SYS__sysctl, &args) == -1) {
    // fallback to reading /proc
    FILE * fp;
    int constexpr MAX_BUFF_SIZE = 256;
    char buffer[MAX_BUFF_SIZE] = {0, };
    fp = std::fopen("/proc/sys/vm/max_map_count", "r");
    vm_cnt_sz = std::fread(buffer, 1, MAX_BUFF_SIZE, fp);
    vm_cnt = std::stoi(buffer, nullptr);
  }
#endif
  return vm_cnt;
}

static void *file_map(const char *path, size_t *length, bool read_ahead) {
  int fd = -1;
  struct stat s;
  void *p = nullptr;
  int flags = MAP_PRIVATE;
  if (read_ahead) {
#if !defined(__AARCH64_GNU__)
    flags |= MAP_POPULATE;
#endif
  }

  if ((fd = open(path, O_RDONLY)) < 0) {
    goto fail;
  }

  if (fstat(fd, &s) < 0) {
    goto fail;
  }

  *length = (size_t)s.st_size;

  if ((p = mmap(nullptr, *length, PROT_READ, flags, fd, 0)) == MAP_FAILED) {
    p = nullptr;
    goto fail;
  }

fail:
  if (p == nullptr) {
    DALI_FAIL("File mapping failed: " + path);
  }
  if (fd > -1) {
    close(fd);
  }
  return p;
}

namespace dali {

using MappedFile = std::tuple<std::weak_ptr<void>, size_t>;

// limit to half of allowed mmaped files
static const unsigned int dali_max_mv_cnt = get_max_vm_cnt() / 2;
// number of currenlty reserved file mappings
static unsigned int dali_reserved_mv_cnt = 0;

// Cache the already opened files: this avoids mapping the same file N times in memory.
// Doing so is wasteful since the file is read-only: we can share the underlying buffer,
// with different pos_.
std::mutex mapped_files_mutex;
std::map<std::string, MappedFile> mapped_files;

LocalFileStream::LocalFileStream(const std::string& path, bool read_ahead) :
  FileStream(path), length_(0), pos_(0), read_ahead_whole_file_(read_ahead) {
  std::lock_guard<std::mutex> lock(mapped_files_mutex);
  std::weak_ptr<void> mapped_memory;
  std::tie(mapped_memory, length_) = mapped_files[path];

  if (!(p_ = mapped_memory.lock())) {
    void *p = file_map(path.c_str(), &length_, read_ahead_whole_file_);
    size_t length_tmp = length_;
    p_ = shared_ptr<void>(p, [=](void*) {
      // we are not touching mapped_files, weak_ptr is enough to check if
      // memory is valid or not
      munmap(p, length_tmp);
     });
    mapped_files[path] = std::make_tuple(p_, length_);
  }

  path_ = path;

  DALI_ENFORCE(p_ != nullptr, "Could not open file " + path + ": " + std::strerror(errno));
}

void LocalFileStream::Close() {
  // Not doing any munmap right now, since Buffer objects might still
  // reference the memory range of the mapping.
  // When last instance of p_ in  LocalFileStream or in memory obtained from
  // LocalFileStream::Get cease to exist memory will be unmapped
  p_ = nullptr;
  length_ = 0;
  pos_ = 0;
}

inline uint8_t* ReadAheadHelper(std::shared_ptr<void> &p, size_t &pos,
                                 size_t &n_bytes, bool read_ahead) {
  auto tmp = static_cast<uint8_t*>(p.get()) + pos;
  // Ask OS to load memory content to RAM to avoid sluggish page fault during actual access to
  // mmaped memory
  if (read_ahead) {
#if !defined(__AARCH64_GNU__)
    madvise(tmp, n_bytes, MADV_WILLNEED);
#endif
  }
  return tmp;
}

void LocalFileStream::Seek(int64 pos) {
  DALI_ENFORCE(pos >= 0 && pos < (int64)length_, "Invalid seek");
  pos_ = pos;
}

// This method saves a memcpy
shared_ptr<void> LocalFileStream::Get(size_t n_bytes) {
  if (pos_ + n_bytes > length_) {
    return nullptr;
  }
  auto tmp = p_;
  shared_ptr<void> p(ReadAheadHelper(p_, pos_, n_bytes, !read_ahead_whole_file_),
    [tmp](void*) {
    // This is an empty lambda, which is a custom deleter for
    // std::shared_ptr.
    // While instantiating shared_ptr, also lambda is instantiated,
    // making a copy of p_. This way, reference counter
    // of p_ is incremented. Therefore, for the duration
    // of life cycle of underlying memory in shared_ptr, file that was
    // mapped creating p_ won't be unmapped
    // It will be freed, when last shared_ptr is deleted.
  });
  pos_ += n_bytes;
  return p;
}

size_t LocalFileStream::Read(uint8_t * buffer, size_t n_bytes) {
  n_bytes = std::min(n_bytes, length_ - pos_);
  memcpy(buffer, ReadAheadHelper(p_, pos_, n_bytes, !read_ahead_whole_file_), n_bytes);
  pos_ += n_bytes;
  return n_bytes;
}

size_t LocalFileStream::Size() const {
  return length_;
}

bool LocalFileStream::ReserveFileMappings(unsigned int num) {
  if (num + dali_reserved_mv_cnt > dali_max_mv_cnt) {
    return false;
  } else {
    dali_reserved_mv_cnt += num;
    return true;
  }
}

void LocalFileStream::FreeFileMappings(unsigned int num) {
  DALI_ENFORCE(dali_reserved_mv_cnt >= num,
      "Trying to free more of mmap reservations than was reserved");
  dali_reserved_mv_cnt -= num;
}

}  // namespace dali
