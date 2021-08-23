// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include <libtar.h>
#include <cstring>
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

constexpr size_t kBlockSize = T_BLOCKSIZE;
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

inline void Unregister(int instance_handle_) {
  instances[instance_handle_] = nullptr;
}

inline TAR* ToTarHandle(void* handle) {
  return reinterpret_cast<TAR*>(handle);
}

inline TAR** ToTarHandle(void** handle) {
  return reinterpret_cast<TAR**>(handle);
}

}  // namespace

ssize_t LibtarReadTarArchive(int instance_handle_, void* buf, size_t count) {
  const auto current_archive = instances[instance_handle_];
  const ssize_t num_read = current_archive->stream_->Read(reinterpret_cast<uint8_t*>(buf), count);
  current_archive->archiveoffset_ += num_read;
  return num_read;
}

int LibtarOpenTarArchive(const char*, int oflags, ...) {
  va_list args;
  va_start(args, oflags);
  const int instance_handle_ = va_arg(args, int);
  va_end(args);
  return instance_handle_;
}

static tartype_t kTarArchiveType = {LibtarOpenTarArchive, [](int) -> int { return 0; },
                                    LibtarReadTarArchive,
                                    [](int, const void*, size_t) -> ssize_t { return 0; }};

TarArchive::TarArchive(std::unique_ptr<FileStream> stream_, size_t offset)
    : stream_(std::move(stream_)),
      archiveoffset_(0),
      instance_handle_(Register(this)),
      eof_(false) {
  tar_open(ToTarHandle(&handle_), "", &kTarArchiveType, 0, instance_handle_, TAR_GNU);
  this->stream_->Seek(0);
  Skip(archiveoffset_);
  ParseHeader();
}

TarArchive::TarArchive(TarArchive&& other) {
  *this = std::move(other);
}

TarArchive::~TarArchive() {
  Invalidate();
}

TarArchive& TarArchive::operator=(TarArchive&& other) {
  if (&other != this) {
    stream_ = std::move(other.stream_);
    std::swap(handle_, other.handle_);
    std::swap(filename_, other.filename_);
    std::swap(filesize_, other.filesize_);
    std::swap(readoffset_, other.readoffset_);
    std::swap(archiveoffset_, other.archiveoffset_);
    std::swap(eof_, other.eof_);
    std::swap(instance_handle_, other.instance_handle_);
    if (instance_handle_ >= 0) {
      std::lock_guard<std::mutex> instances_lock(instances_mutex);
      instances[instance_handle_] = this;
    }
    other.Invalidate();
  }
  return *this;
}

constexpr size_t RoundToBlockSize(size_t count) {
  return align_up(count, kBlockSize);
}

bool TarArchive::NextFile() {
  if (eof_) {
    return false;
  }
  Skip(RoundToBlockSize(filesize_) - readoffset_);
  if (eof_) {
    return false;
  }
  eof_ = ParseHeader();
  return !eof_;
}

bool TarArchive::EndOfArchive() const {
  return eof_;
}

const std::string& TarArchive::GetFileName() const {
  return filename_;
}

size_t TarArchive::GetFileSize() const {
  return filesize_;
}

TarArchive::EntryType TarArchive::GetFileType() const {
  return filetype_;
}

std::shared_ptr<void> TarArchive::ReadFile() {
  archiveoffset_ -= readoffset_;
  size_t old_readoffset = readoffset_;
  readoffset_ = 0;
  stream_->Seek(archiveoffset_);
  auto out = stream_->Get(filesize_);
  if (out != nullptr) {
    archiveoffset_ += filesize_;
    readoffset_ = filesize_;
  } else {
    archiveoffset_ += old_readoffset;
    readoffset_ = old_readoffset;
  }
  return out;
}

size_t TarArchive::Read(uint8_t* buffer, size_t count) {
  if (eof_) {
    return 0;
  }
  count = clamp(filesize_ - readoffset_, 0_u64, count);
  size_t num_read_bytes = stream_->Read(buffer, count);
  readoffset_ += num_read_bytes;
  archiveoffset_ += num_read_bytes;
  return num_read_bytes;
}

bool TarArchive::EndOfFile() const {
  return readoffset_ >= filesize_;
}

inline void TarArchive::Skip(size_t count) {
  if ((archiveoffset_ += count) >= stream_->Size()) {
    Invalidate();
    return;
  }
  stream_->Seek(archiveoffset_);
  readoffset_ += count;
}

inline bool TarArchive::ParseHeader() {
  int errorcode = th_read(ToTarHandle(handle_));
  if (errorcode) {
    DALI_ENFORCE(errorcode != -1,
                 (std::string) "Corrupted tar file at " + ToTarHandle(handle_)->pathname);
    Invalidate();
  } else {
    filename_ = th_get_pathname(ToTarHandle(handle_));
    filesize_ = th_get_size(ToTarHandle(handle_));

    if (TH_ISREG(ToTarHandle(handle_))) {
      filetype_ = ENTRY_FILE;
    } else if (TH_ISDIR(ToTarHandle(handle_))) {
      filetype_ = ENTRY_DIR;
    } else if (TH_ISLNK(ToTarHandle(handle_))) {
      filetype_ = ENTRY_HARDLINK;
    } else if (TH_ISSYM(ToTarHandle(handle_))) {
      filetype_ = ENTRY_SYMLINK;
    } else if (TH_ISCHR(ToTarHandle(handle_))) {
      filetype_ = ENTRY_CHARDEV;
    } else if (TH_ISBLK(ToTarHandle(handle_))) {
      filetype_ = ENTRY_BLOCKDEV;
    } else {
      DALI_ENFORCE(TH_ISFIFO(ToTarHandle(handle_)), "Entry type in a tar archive not recognized");
      filetype_ = ENTRY_FIFO;
    }
  }
  readoffset_ = 0;
  return errorcode;
}

void TarArchive::Invalidate() {
  stream_.reset();
  if (handle_ != nullptr) {
    tar_close(ToTarHandle(handle_));
    handle_ = nullptr;
  }
  filename_ = "";
  filesize_ = 0;
  readoffset_ = 0;
  archiveoffset_ = 0;
  eof_ = true;
  if (instance_handle_ >= 0) {
    Unregister(instance_handle_);
  }
  instance_handle_ = -1;
}


}  // namespace detail
}  // namespace dali
