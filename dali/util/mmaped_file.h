// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
  size_t Read(uint8_t * buffer, size_t n_bytes) override;
  void Seek(int64 pos) override;
  int64 Tell() const override;
  size_t Size() const override;

  ~MmapedFileStream() override {
    Close();
  }

 private:
  std::shared_ptr<void> p_;
  size_t length_;
  size_t pos_;
  bool read_ahead_whole_file_;
};

}  // namespace dali

#endif  // DALI_UTIL_MMAPED_FILE_H_
