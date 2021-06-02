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

#ifndef DALI_UTIL_CUFILE_H_
#define DALI_UTIL_CUFILE_H_

#include <cstdio>
#include <memory>
#include <string>

#include "dali/core/api_helper.h"
#include "dali/core/common.h"
#include "dali/util/file.h"

namespace dali {

class DLL_PUBLIC CUFileStream : public FileStream {
 public:
  // for compatibility with FileStream API
  class MappingReserver {
   public:
    explicit MappingReserver(unsigned int num) {}

    MappingReserver() {}

    bool CanShareMappedData() {
      return false;
    }
  };

  static std::unique_ptr<CUFileStream> Open(const std::string& uri, bool read_ahead, bool use_mmap);
  /*
   * It accepts the base address of the buffer to read to and the offset in it
   * The API is the effect how cufile works - it need to get the base address of the registered
   * buffer and the offset where it should put the data.
   */
  virtual size_t ReadGPU(uint8_t* buffer, size_t n_bytes, size_t offset = 0) = 0;
  virtual size_t ReadGPUImpl(uint8_t* buffer, size_t n_bytes,
                             size_t buffer_offset, size_t file_offset) = 0;

 protected:
  explicit CUFileStream(const std::string& path) : FileStream(path) {}
};

}  // namespace dali

#endif  // DALI_UTIL_CUFILE_H_
