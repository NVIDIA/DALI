// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  /**
   * @brief Reads `n_bytes` to the buffer at position `offset`
   *
   * It accepts the base address of the buffer to read to and the offset in it
   * The API is the effect how cufile works - it need to get the base address of the registered
   * buffer and the offset where it should put the data.
   */
  virtual size_t ReadGPU(void *buffer, size_t n_bytes, ptrdiff_t buffer_offset = 0) = 0;

  /**
   * @brief Reads `n_bytes` to the buffer at position `offset` from a given position in the file.
   *
   * The file_offset is absolute - the function neither depends on or affects the file pointer.
   * This function is thread-safe.
   */
  virtual size_t ReadAtGPU(void *buffer, size_t n_bytes,
                           ptrdiff_t buffer_offset, int64 file_offset) = 0;

 protected:
  explicit CUFileStream(const std::string& path) : FileStream(path) {}
};

}  // namespace dali

#endif  // DALI_UTIL_CUFILE_H_
