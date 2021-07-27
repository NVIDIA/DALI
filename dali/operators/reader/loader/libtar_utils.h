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

#ifndef DALI_OPERATORS_READER_LOADER_LIBTAR_UTILS_H_
#define DALI_OPERATORS_READER_LOADER_LIBTAR_UTILS_H_

#include <libtar.h>

#include <cstddef>
#include <iterator>
#include <mutex>
#include <string>

#include "dali/core/common.h"

namespace dali {
namespace detail {

class TarArchive {
 public:
  explicit TarArchive(const std::string& filepath);
  TarArchive(TarArchive&&);
  ~TarArchive();

  bool NextFile();
  bool IsAtFile();

  std::string GetFileName() const;
  uint64 GetFileSize() const;

  std::string Read();
  std::string Read(uint64 count);
  bool Eof() const;

 private:
  TAR* handle;
  std::string filename;
  uint64 filesize;
  uint64 readsize;

  void InitNextFile();
  void TryClose();
  void TryLibtar(int, std::string&&);
};

}  // namespace detail
}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_LIBTAR_UTILS_H_
