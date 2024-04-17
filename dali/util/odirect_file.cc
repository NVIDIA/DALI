// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "dali/core/util.h"
#include "dali/util/odirect_file.h"
#include "dali/core/error_handling.h"

namespace dali {

static constexpr size_t kODirectAlignment = 4096;
static constexpr size_t kODirectChunkSize = 2 << 20;  // 2M

static size_t GetODirectEnvVals(const char *env_name, size_t min_value, size_t default_value,
                                bool pow2 = true, size_t alignment = 1) {
  char *env = getenv(env_name);
  int len = 0;
  if (env && (len = strlen(env))) {
    for (int i = 0; i < len; i++) {
      bool valid = std::isdigit(env[i]) || (i == len - 1 && (env[i] == 'k' || env[i] == 'M'));
      if (!valid) {
        DALI_FAIL(make_string(
          env_name, " must be a number, optionally followed by 'k' or 'M', got: ",
          env));
      }
    }
    size_t s = atoll(env);
    if (env[len-1] == 'k')
      s <<= 10;
    else if (env[len-1] == 'M')
      s <<= 20;
    DALI_ENFORCE(!pow2 || is_pow2(s), make_string(env_name, " must be a power of two, got ", s));
    DALI_ENFORCE(pow2 || alignment_offset(s, alignment) == 0,
                 make_string(env_name, " must be a aligned to ", min_value, ", got ", s));
    DALI_ENFORCE(s >= min_value && s <= (16 << 20),
      make_string(env_name, " be a power of two between ",
                  min_value, " and 16 M, got: ", s));
    return s;
  } else {
    // not set or empty
    return default_value;
  }
}

ODirectFileStream::ODirectFileStream(const std::string& path) : FileStream(path) {
  fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
  DALI_ENFORCE(fd_ >= 0, "Could not open file " + path + ": " + std::strerror(errno));
}

size_t ODirectFileStream::GetAlignment() {
  return GetODirectEnvVals("DALI_ODIRECT_ALIGNMENT", kODirectAlignment, kODirectAlignment, true);
}

size_t ODirectFileStream::GetLenAlignment() {
  return GetODirectEnvVals("DALI_ODIRECT_LEN_ALIGNMENT", kODirectAlignment, kODirectAlignment,
                           true);
}

size_t ODirectFileStream::GetChunkSize() {
  return GetODirectEnvVals("DALI_ODIRECT_CHUNK_SIZE", kODirectAlignment, kODirectChunkSize, false,
                           ODirectFileStream::GetLenAlignment() );
}

ODirectFileStream::~ODirectFileStream() {
  Close();
}

void ODirectFileStream::Close() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void ODirectFileStream::SeekRead(ptrdiff_t pos, int whence) {
  DALI_ENFORCE(lseek(fd_, pos, whence) >= 0, make_string(
               "Seek operation failed: ", std::strerror(errno)));
}

ptrdiff_t ODirectFileStream::TellRead() const {
  return lseek(fd_, 0, SEEK_CUR);
}

size_t ODirectFileStream::ReadAt(void * buffer, size_t n_bytes, off_t offset) {
  size_t n_read = pread(fd_, buffer, n_bytes, offset);
  return n_read;
}

size_t ODirectFileStream::Read(void *buffer, size_t n_bytes) {
  size_t n_read = read(fd_, buffer, n_bytes);
  return n_read;
}

size_t ODirectFileStream::Size() const {
  struct stat sb;
  if (stat(path_.c_str(), &sb) == -1) {
    DALI_FAIL("Unable to stat file " + path_ + ": " + std::strerror(errno));
  }
  return sb.st_size;
}

}  // namespace dali
