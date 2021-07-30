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
#include <list>
#include <string>
#include <tuple>
#include <utility>
#include "dali/core/error_handling.h"
#include "dali/core/math_util.h"
#include "dali/core/util.h"

namespace dali {
namespace detail {

namespace {

constexpr uint64_t kBlockSize = T_BLOCKSIZE;
static_assert(is_pow2(kBlockSize),
              "The implementation assumes that the block size is a power of 2");

constexpr uint64_t kEmptyEofBlocks = 2;
constexpr uint64_t kTarArchiveBufferInitSize = 1;

std::mutex instances_mutex;
std::list<std::vector<TarArchive*>> instances_registry = {
    std::vector<TarArchive*>(kTarArchiveBufferInitSize)};
TarArchive** instances = instances_registry.back().data();

int Register(TarArchive* archive) {
  std::lock_guard<std::mutex> instances_lock(instances_mutex);
  for (auto& instances_entry : instances_registry.back()) {
    if (instances_entry != nullptr) {
      continue;
    }
    instances_entry = archive;
    return &instances_entry - instances;
  }
  std::vector<TarArchive*>& old = instances_registry.back();
  instances_registry.emplace_back();
  std::vector<TarArchive*>& curr = instances_registry.back();
  curr.reserve(old.size() * 2);
  curr = old;
  curr.push_back(archive);
  curr.resize(old.size() * 2, nullptr);
  instances = curr.data();
  return old.size();
}

inline void Unregister(int instance_handle) {
  instances[instance_handle] = nullptr;
}

}  // namespace

TarArchive::TarArchive(std::unique_ptr<FileStream> stream)
    : stream(std::move(stream)), archiveoffset(0), instance_handle(Register(this)), eof(false) {
  this->stream->Seek(0);
  ParseHeader();
}

TarArchive::TarArchive(TarArchive&& other) {
  *this = static_cast<TarArchive&&>(other);
}

TarArchive::~TarArchive() {
  Unregister(instance_handle);
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

uint64_t RoundToBlockSize(uint64_t count) {
  return align_up(count, kBlockSize);
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

uint64_t TarArchive::GetFileSize() const {
  return filesize;
}

std::shared_ptr<void> TarArchive::ReadFile() {
  archiveoffset -= readoffset;
  readoffset = 0;
  stream->Seek(archiveoffset);
  auto out = stream->Get(filesize);
  if (out != nullptr) {
    archiveoffset += filesize;
    readoffset = filesize;
  }
  return out;
}

size_t TarArchive::Read(uint8_t* buffer, size_t count) {
  if (eof) {
    return 0;
  }
  count = clamp(filesize - readoffset, 0_u64, count);
  uint64_t num_read_bytes = stream->Read(buffer, count);
  readoffset += num_read_bytes;
  archiveoffset += num_read_bytes;
  return num_read_bytes;
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
      DALI_FAIL("Corrupted tar file at " + handle->pathname);
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
