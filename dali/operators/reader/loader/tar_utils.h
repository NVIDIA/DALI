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

#ifndef DALI_OPERATORS_READER_LOADER_TAR_UTILS_H_
#define DALI_OPERATORS_READER_LOADER_TAR_UTILS_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/util/file.h"

namespace dali {
namespace detail {
class TarArchive {
 public:
  TarArchive() = default;
  explicit TarArchive(std::unique_ptr<FileStream> stream);
  TarArchive(TarArchive&&);
  ~TarArchive();
  TarArchive& operator=(TarArchive&&);

  bool NextFile();
  bool IsAtFile() const;

  std::string GetFileName() const;
  uint64_t GetFileSize() const;

  std::shared_ptr<void> ReadFile();
  size_t Read(uint8_t* buffer, size_t count);
  bool Eof() const;

 private:
  std::unique_ptr<FileStream> stream;
  std::string filename;
  uint64_t filesize = 0;
  uint64_t readoffset = 0;
  uint64_t archiveoffset = 0;
  int instance_handle = -1;
  bool eof = true;
  bool ParseHeader();
  void Skip(int64_t count);
};

}  // namespace detail
}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_TAR_UTILS_H_
