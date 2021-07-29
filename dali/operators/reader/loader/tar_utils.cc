// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/reader/loader/tar_utils.h"
#include <libtar.h>
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include "dali/core/error_handling.h"

namespace dali {
namespace detail {

constexpr uint64 kBlockSize = 512;
constexpr uint64 kEmptyEofBlocks = 2;

std::mutex instances_mutex;
vector<TarArchive*> instances;

int Register(TarArchive* archive) {
  std::lock_guard<std::mutex> instances_lock(instances_mutex);
  int instance_handle = -1;
  for (auto& instances_entry : instances) {
    if (instances_entry != nullptr) {
      continue;
    }
    instances_entry = archive;
    instance_handle = &instances_entry - instances.data();
  }
  if (instance_handle == -1) {
    instance_handle = instances.size();
    instances.push_back(archive);
  }
  return instance_handle;
}

TarArchive::TarArchive(std::unique_ptr<FileStream>&& stream)
    : stream(std::forward<std::unique_ptr<FileStream>>(stream)),
      eof(false),
      instance_handle(Register(this)),
      archiveoffset(0) {
  this->stream->Seek(0);
  ParseHeader();
}

TarArchive::TarArchive(TarArchive&& other) {
  *this = std::forward<TarArchive>(other);
}

TarArchive& TarArchive::operator=(TarArchive&& other) {
  stream = std::move(other.stream);
  eof = other.eof;
  filename = other.filename;
  filesize = other.filesize;
  readoffset = other.readoffset;
  archiveoffset = other.archiveoffset;

  std::lock_guard<std::mutex> instances_lock(instances_mutex);
  instances[instance_handle] = nullptr;
  instance_handle = other.instance_handle;
  return *this;
}

TarArchive::~TarArchive() {
  std::lock_guard<std::mutex> instances_lock(instances_mutex);
  instances[instance_handle] = nullptr;
}

uint64 RoundToBlockSize(uint64 count) {
  return ((count + kBlockSize - 1) / kBlockSize) * kBlockSize;
}

bool TarArchive::NextFile() {
  if (eof) {
    return false;
  }
  Skip(RoundToBlockSize(filesize) - readoffset);
  eof = ParseHeader();
  return !eof;
}

bool TarArchive::IsAtFile() const {
  return !eof;
}

std::string TarArchive::GetFileName() const {
  return filename;
}

uint64 TarArchive::GetFileSize() const {
  return filesize;
}

std::vector<uint8_t> TarArchive::Read() {
  return Read(filesize - readoffset);
}

std::vector<uint8_t> TarArchive::Read(uint64 count) {
  if (eof) {
    return vector<uint8_t>();
  }
  count = std::max(std::min(count, filesize - readoffset), static_cast<uint64>(0));
  std::vector<uint8_t> out(count);
  uint64 num_read_bytes = stream->Read(out.data(), count);
  readoffset += num_read_bytes;
  archiveoffset += num_read_bytes;
  return out;
}

bool TarArchive::Eof() const {
  return readoffset >= filesize;
}

inline void TarArchive::Skip(int64 count) {
  stream->Seek(archiveoffset += count);
  readoffset += count;
}

int LibtarOpenTarArchive(const char*, int oflags, ...) {
  va_list args;
  va_start(args, oflags);
  const int instance_handle = va_arg(args, int);
  va_end(args);
  return instance_handle;
}


inline bool TarArchive::ParseHeader() {
  TAR* handle;
  tartype_t type = {LibtarOpenTarArchive, [](int) -> int { return 0; },
                    [](int instance_handle, void* buf, size_t count) -> ssize_t {
                      const auto current_archive = instances[instance_handle];
                      const ssize_t num_read =
                          current_archive->stream->Read(reinterpret_cast<uint8_t*>(buf), count);
                      current_archive->archiveoffset += num_read;
                      return num_read;
                    },
                    [](int, const void*, size_t) -> ssize_t { return 0; }};
  tar_open(&handle, "", &type, 0, instance_handle, TAR_GNU);

  int errorcode = th_read(handle);
  if (errorcode) {
    filename = "";
    filesize = 0;
    if (errorcode == -1) {
      DALI_FAIL(R"(Corrupted tar file at )" + handle->pathname);
    }
  } else {
    filename = th_get_pathname(handle);
    filesize = th_get_size(handle);
  }
  readoffset = 0;

  tar_close(handle);
  return errorcode;
}


}  // namespace detail
}  // namespace dali
