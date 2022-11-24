// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_MMAPED_FILE_H_
#define DALI_UTIL_MMAPED_FILE_H_

#include <cstdio>
#include <string>
#include <memory>

#include "dali/core/common.h"
#include "dali/util/file.h"

namespace dali {

class MmapedFileStream : public FileStream {
 public:
  explicit MmapedFileStream(const std::string& path, bool read_ahead);
  void Close() override;
  shared_ptr<void> Get(size_t n_bytes) override;
  static bool ReserveFileMappings(unsigned int num);
  static void FreeFileMappings(unsigned int num);
  size_t Read(void *buffer, size_t n_bytes) override;
  void SeekRead(ptrdiff_t pos, int whence = SEEK_SET) override;
  int64 TellRead() const override;
  size_t Size() const override;

  ~MmapedFileStream() override;

 private:
  MmapedFileStream(const MmapedFileStream &) = delete;
  MmapedFileStream(MmapedFileStream &&) = delete;
  std::shared_ptr<void> p_;
  size_t length_;
  size_t pos_;
  bool read_ahead_whole_file_;
};

}  // namespace dali

#endif  // DALI_UTIL_MMAPED_FILE_H_
