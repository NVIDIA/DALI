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

#ifndef DALI_UTIL_FILE_H_
#define DALI_UTIL_FILE_H_

#include <cstdio>
#include <string>

#include "dali/api_helper.h"
#include "dali/common.h"

namespace dali {

class DLL_PUBLIC FileStream {
 public:
  static FileStream * Open(const std::string& uri);

  virtual void Close() = 0;
  virtual size_t Read(uint8_t * buffer, size_t n_bytes) = 0;
  virtual void Seek(int64 pos) = 0;
  virtual size_t Size() const = 0;
  virtual ~FileStream() {}

 protected:
  explicit FileStream(const std::string& path) :
    path_(path)
    {}

  std::string path_;
};

}  // namespace dali

#endif  // DALI_UTIL_FILE_H_
