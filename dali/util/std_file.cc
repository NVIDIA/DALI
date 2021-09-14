// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "dali/core/error_handling.h"
#include "dali/util/std_file.h"

namespace dali {

StdFileStream::StdFileStream(const std::string& path) : FileStream(path) {
  fp_ = std::fopen(path.c_str(), "rb");
  DALI_ENFORCE(fp_ != nullptr, "Could not open file " + path + ": " + std::strerror(errno));
}

void StdFileStream::Close() {
  if (fp_ != nullptr) {
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

void StdFileStream::Seek(int64 pos) {
  DALI_ENFORCE(!std::fseek(fp_, pos, SEEK_SET),
               "Seek operation did not succeed: " + std::string(std::strerror(errno)));
}

int64 StdFileStream::Tell() const {
  return std::ftell(fp_);
}

size_t StdFileStream::Read(uint8_t* buffer, size_t n_bytes) {
  size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
  return n_read;
}

shared_ptr<void> StdFileStream::Get(size_t /*n_bytes*/) {
  // this unction should return a pointer inside mmaped file
  // it doesn't make sense in case of StdFileStream
  return {};
}

size_t StdFileStream::Size() const {
  struct stat sb;
  if (stat(path_.c_str(), &sb) == -1) {
    DALI_FAIL("Unable to stat file " + path_ + ": " + std::strerror(errno));
  }
  return sb.st_size;
}

}  // namespace dali
