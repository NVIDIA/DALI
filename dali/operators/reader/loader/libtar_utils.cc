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
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <future>
#include <string>
#include <utility>

#include <iostream>

#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/filesystem.h"

namespace dali {
namespace detail {

// Constructors
TarArchive::TarArchive(const std::string& filepath) {
  TryLibtar(tar_open(&handle, filepath.c_str(), nullptr, O_RDONLY, 0, TAR_GNU),
            "Could not open tha tar archive at");
  InitNextFile();
}

TarArchive::TarArchive(TarArchive&& other)
    : handle(other.handle),
      filename(other.filename),
      filesize(other.filesize),
      leftovers(other.leftovers),
      readsize(other.readsize) {
  other.handle = nullptr;
}

TarArchive::~TarArchive() {
  TryClose();
}

// Archive Access Methods


bool TarArchive::NextFile() {
  if (!IsAtFile()) {
    return false;
  }

  TryLibtar(lseek(handle->fd, filesize - readsize, SEEK_CUR), "Could not skip to the next tar registry at");
  InitNextFile();
  return IsAtFile();
}

inline bool TarArchive::IsAtFile() {
  return handle != nullptr;
}

// File Access Methods
std::string TarArchive::GetFileName() const {
  return filename;
}

uint64 TarArchive::GetFileSize() const {
  return filesize;
}

std::string TarArchive::Read() {
  const uint64 leftover_size = leftovers.size();
  const uint64 blocks =
      ((filesize - readsize - leftover_size) + T_BLOCKSIZE - 1) / T_BLOCKSIZE;  // for rounding up

  std::string out(std::move(leftovers));
  leftovers.clear();

  out.resize(filesize - readsize);
  out.reserve(leftover_size + blocks * T_BLOCKSIZE);
  handle->type->readfunc(handle->fd, (char*)(out.data() + leftover_size), blocks * T_BLOCKSIZE);
  readsize += out.size();
  return out;
}

std::string TarArchive::Read(uint64 count) {
  count = std::min(count, filesize - readsize);
  const uint64 leftover_size = leftovers.size();
  const uint64 blocks =
      ((count - readsize - leftover_size) + T_BLOCKSIZE - 1) / T_BLOCKSIZE;  // for rounding up

  std::string out(std::move(leftovers));

  out.resize(leftover_size + blocks * T_BLOCKSIZE);
  for (uint64 i = leftover_size; i < count; i += T_BLOCKSIZE) {
    tar_block_read(handle, out.data() + i);
  }

  readsize += count;

  leftovers = std::move(out.substr(count, readsize));
  out.resize(count);

  return out;
}

bool TarArchive::Eof() const {
  return readsize == filesize;
}

// Private Helper Functions
inline void TarArchive::InitNextFile() {
  int errorcode = th_read(handle);
  if (errorcode) {
    std::cerr << "ErrorCode for th_read: " << errorcode << std::endl;
    TryClose();
    return;
  }
  TryLibtar(!TH_ISREG(handle), "Found a non-file entry at");

  filename = th_get_pathname(handle);
  filesize = th_get_size(handle);
  readsize = 0;
}

inline void TarArchive::TryClose() {
  if (handle) {
    std::cerr << "The handle is getting closed" << std::endl;
    TryLibtar(tar_close(handle), "Could not close the tar archive at");
    handle = nullptr;
  }
}

void TarArchive::TryLibtar(int exitcode, std::string&& error_msg) {
  if (exitcode) {
    if (handle) {
      DALI_ERROR(error_msg + " " + handle->pathname);
      handle = nullptr;
    } else {
      DALI_ERROR("Could not access a tar archive");
    }
  }
}

}  // namespace detail
}  // namespace dali
