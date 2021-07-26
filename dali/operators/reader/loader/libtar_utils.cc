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

#include "dali/operators/reader/loader/libtar_utils.h"

#include <fcntl.h>

#include <algorithm>
#include <cstdlib>
#include <future>
#include <string>
#include <utility>

#include "dali/core/error_handling.h"

namespace dali {
namespace detail {

TarArchive::TarArchive(const std::string& filepath) {
  if (tar_open(&handle, filepath.c_str(), nullptr, O_RDONLY, 0, TAR_GNU)) {
    std::string error = "Could not open the tar archive at ";
    error += filepath;
    DALI_ERROR(error);
    handle = nullptr;
  }
}

TarArchive::TarArchive(TarArchive&& other) : handle(other.handle) {
  other.handle = nullptr;
}

TarArchive::~TarArchive() {
  TryClose();
}

bool TarArchive::Next() {
  if (CheckEnd()) {
    return;
  }

  if (tar_skip_regfile(handle)) {
    TryClose();
  }

  buffer_init = false;

  return !CheckEnd();
}

inline bool TarArchive::CheckEnd() {
  return handle == nullptr;
}

void TarArchive::TryClose() {
  if (handle && tar_close(handle)) {
    std::string error = "Could not close the tar archive at ";
    error += handle->pathname;
    DALI_ERROR(error);
  }
  handle = nullptr;
}

std::string TarArchive::FileName() {
  return CheckEnd() ? "" : th_get_pathname(handle);
}

std::string TarArchive::FileRead(uint64 count) {
  if (CheckEnd()) {
    return "";
  }

  std::string out;
  BufferTryInit();
  while (buffer_size == T_BLOCKSIZE && count) {
    count -= BufferRead(out, count);
    if (buffer_offset == T_BLOCKSIZE) {
      BufferUpdate();
    }
  }

  return std::move(out);
}

bool TarArchive::Eof() {
  if (CheckEnd()) {
    return true;
  }

  BufferTryInit();
  return buffer_size == buffer_offset;
}

inline void TarArchive::BufferTryInit() {
  if (!buffer_init) {
    BufferUpdate();
    buffer_init = true;
  }
}

inline uint64 TarArchive::BufferRead(std::string& out, uint64 count) {
  count = std::min(count, static_cast<uint64>(buffer_size - buffer_offset));
  out.append(buffer + buffer_offset, count);
  buffer_offset += count;
}

inline void TarArchive::BufferUpdate() {
  buffer_offset = 0;
  buffer_size = tar_block_read(handle, buffer);
}

}  // namespace detail
}  // namespace dali
