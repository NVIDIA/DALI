// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_ODIRECT_FILE_H_
#define DALI_UTIL_ODIRECT_FILE_H_

#include <cstdio>
#include <string>
#include <memory>

#include "dali/core/common.h"
#include "dali/util/file.h"

namespace dali {

class DLL_PUBLIC ODirectFileStream : public FileStream {
 public:
  explicit ODirectFileStream(const std::string& path);
  void Close() override;
  shared_ptr<void> Get(size_t n_bytes) override;
  size_t Read(void * buffer, size_t n_bytes) override;
  size_t ReadAt(void * buffer, size_t n_bytes, off_t offset);
  static size_t GetAlignment();
  static size_t GetLenAlignment();
  static size_t GetChunkSize();
  void SeekRead(ptrdiff_t pos, int whence = SEEK_SET) override;
  ptrdiff_t TellRead() const override;
  size_t Size() const override;

  ~ODirectFileStream() override;

 private:
  int fd_;
};

}  // namespace dali

#endif  // DALI_UTIL_ODIRECT_FILE_H_
