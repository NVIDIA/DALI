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

#ifndef DALI_UTIL_FILE_H_
#define DALI_UTIL_FILE_H_

#include <cstdio>
#include <streambuf>
#include <memory>
#include <string>
#include <optional>
#include "dali/core/api_helper.h"
#include "dali/core/common.h"
#include "dali/core/stream.h"
#include "dali/core/format.h"

namespace dali {

class DLL_PUBLIC FileStream : public InputStream {
 public:
  class MappingReserver {
   public:
    explicit MappingReserver(unsigned int num) : reserved(0) {
      if (FileStream::ReserveFileMappings(num)) {
        reserved = num;
      }
    }

    MappingReserver() : MappingReserver(0) {}

    MappingReserver(const MappingReserver &) = delete;
    MappingReserver &operator=(const MappingReserver &) = delete;

    MappingReserver(MappingReserver &&other) : MappingReserver(other.reserved) {
      other.reserved = 0;
    }

    MappingReserver &operator=(MappingReserver &&other) {
      reserved = other.reserved;
      other.reserved = 0;
      return *this;
    }

    MappingReserver &operator=(MappingReserver &other) {
      reserved = other.reserved;
      other.reserved = 0;
      return *this;
    }

    bool CanShareMappedData() {
      return reserved != 0;
    }

    ~MappingReserver() {
      if (reserved) {
        FileStream::FreeFileMappings(reserved);
      }
    }

   private:
    unsigned int reserved;
  };

  struct Options {
    bool read_ahead;
    bool use_mmap;
    bool use_odirect;
  };

  /**
   * @brief Opens file stream
   *
   * @param uri URI to open
   * @param opts options
   * @param size If provided, we can defer the actual reading of the stream until it needs to be
   * read (e.g. especially useful for remote storage)
   * @return std::unique_ptr<FileStream>
   */
  static std::unique_ptr<FileStream> Open(const std::string &uri,
                                          Options opts = {false, false, false},
                                          std::optional<size_t> size = std::nullopt);

  virtual void Close() = 0;
  virtual bool CanMemoryMap() { return false; }
  virtual shared_ptr<void> Get(size_t n_bytes) {
    throw std::logic_error(
        make_string("memory mapping is not supported for this stream type. uri=", path_));
  }
  const std::string& path() const { return path_; }
  virtual ~FileStream() {}

 protected:
  static bool ReserveFileMappings(unsigned int num);
  static void FreeFileMappings(unsigned int num);
  explicit FileStream(const std::string &path) : path_(path) {}

  std::string path_;
};

/**
 * @brief Custom streambuf implementation that reads from FileStream.
 * @remarks It is useful to be used together with std::istream
 */
template <size_t BufferSize = (1 << 10)>
class FileStreamBuf : public std::streambuf {
 public:
  explicit FileStreamBuf(FileStream *reader) : reader_(reader) {
    setg(buffer_, buffer_, buffer_);  // Initialize get area pointers
  }

 protected:
  int_type underflow() override {
    if (gptr() == egptr()) {  // get area is exhausted
      size_t nbytes = reader_->Read(buffer_, BufferSize);
      if (nbytes == 0)
        return traits_type::eof();
      setg(buffer_, buffer_, buffer_ + nbytes);
    }
    return traits_type::to_int_type(*gptr());
  }

 private:
  FileStream *reader_;
  char buffer_[BufferSize];
};

}  // namespace dali

#endif  // DALI_UTIL_FILE_H_
