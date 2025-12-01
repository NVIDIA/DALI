// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#if !defined(__AARCH64_QNX__) && !defined(__AARCH64_GNU__) && !defined(__aarch64__)
#include <linux/sysctl.h>
#include <sys/syscall.h>
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "dali/core/dynlink_cufile.h"
#include "dali/core/format.h"
#include "dali/util/std_cufile.h"

// import cufile and return handle
static void cufile_open(cufile::CUFileHandle& fh, size_t& length, const char* path) {
  // make sure handle is closed
  fh.Close();

  struct stat s;

  // we need to be careful: if a symbolic link was provided,
  // we cannot use O_DIRECT. So better extract the realpath
  std::unique_ptr<char, decltype(&free)> rpath(realpath(path, NULL), &free);
  if (rpath == nullptr) {
    DALI_FAIL("Could not resolve real path of: ", path);
  }

  // do conventional open
  if ((fh.fdd = open(rpath.get(), O_RDONLY | O_DIRECT)) < 0) {
    DALI_FAIL("CUFile open failed: ", path);
  }
  if ((fh.fd = open(rpath.get(), O_RDONLY)) < 0) {
    DALI_FAIL("CUFile open failed: ", path);
  }
  if (fstat(fh.fd, &s) < 0) {
    DALI_FAIL("CUFile stats failed: ", path);
  }

  // get length
  length = static_cast<size_t>(s.st_size);

  // import file for reading
  CUfileDescr_t descr = {};
  memset(&descr, 0, sizeof(CUfileDescr_t));
  descr.handle.fd = fh.fdd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileError_t status = cuFileHandleRegister(&(fh.cufh), &descr);
  if (status.err != CU_FILE_SUCCESS) {
    DALI_FAIL("CUFile import failed: ", path, ". ", cufileop_status_error(status.err), ".");
  }
}

namespace dali {

StdCUFileStream::StdCUFileStream(const std::string& path) : CUFileStream(path) {
  // open file
  cufile_open(f_, length_, path.c_str());
  // set the path to current path
  path_ = path;
}

StdCUFileStream::~StdCUFileStream() {
  Close();
}


void StdCUFileStream::Close() {
  // do not deallocate here, since
  // other processes might still use the handles
  f_.Close();
  length_ = 0;
  pos_ = 0;
  path_ = "";
}

void StdCUFileStream::SeekRead(ptrdiff_t pos, int whence) {
  if (whence == SEEK_CUR)
    pos += pos_;
  else if (whence == SEEK_END)
    pos += length_;
  DALI_ENFORCE(pos >= 0 && pos <= (int64_t)length_, "Invalid seek");
  pos_ = pos;
}

ptrdiff_t StdCUFileStream::TellRead() const {
  return pos_;
}

void StdCUFileStream::HandleIOError(int64_t ret) const {
  if (ret == -1) {
    std::string errmsg(256, '\0');
    int e = errno;
    auto ret = strerror_r(e, &errmsg[0], errmsg.size());
    if (ret != 0) {
      DALI_FAIL("Unknown CUFile error: ", e);
    }
    DALI_FAIL("CUFile read failed for file ", path_, " with error (", e, "): ", errmsg);
  } else {
    DALI_FAIL("CUFile read failed for file ", path_, " with error (", -ret,
              "): ", cufileop_status_error(static_cast<CUfileOpError>(-ret)));
  }
}

size_t StdCUFileStream::ReadAtGPU(void *gpu_buffer, size_t n_bytes,
                                  ptrdiff_t buffer_offset, int64_t file_offset) {
  // compute size
  n_bytes = std::min(n_bytes, length_ - file_offset);

  // read data: backup n_bytes here and create a read-offset
  ssize_t n_read = n_bytes;
  off_t read_off = 0;
  while (n_read > 0) {
    int64_t read = cuFileRead(f_.cufh, gpu_buffer, n_read,
                              file_offset + read_off, buffer_offset);

    if (read >= 0) {
      // worked well, continue
      n_read -= read;
      read_off += read;
      buffer_offset += read;
    } else {
      // say goodbye here
      HandleIOError(read);
    }
  }

  return n_bytes;
}

size_t StdCUFileStream::ReadGPU(void *gpu_buffer, size_t n_bytes, ptrdiff_t buffer_offset) {
  n_bytes = ReadAtGPU(gpu_buffer, n_bytes, buffer_offset, 0);

  // we can safely advance the file pointer here
  pos_ += n_bytes;
  return n_bytes;
}

size_t StdCUFileStream::Read(void *cpu_buffer, size_t n_bytes) {
  // compute size
  n_bytes = std::min(n_bytes, length_ - pos_);

  // read data
  size_t n_read = n_bytes;
  off_t read_off = 0;
  while (n_read) {
    int64_t read = pread(f_.fd, static_cast<char*>(cpu_buffer) + read_off, n_read, pos_ + read_off);

    if (read >= 0) {
      n_read -= read;
      read_off += read;
    } else {
      HandleIOError(read);
    }
  }

  pos_ += n_bytes;
  return n_bytes;
}

size_t StdCUFileStream::Size() const {
  return length_;
}

}  // namespace dali
