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

#ifndef DALI_UTIL_STD_CUFILE_H_
#define DALI_UTIL_STD_CUFILE_H_

#include <cstdio>
#include <string>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/util/cufile.h"
#include "dali/util/cufile_helper.h"


namespace dali {

class StdCUFileStream : public CUFileStream {
 public:
  explicit StdCUFileStream(const std::string& path);
  void Close() override;
  shared_ptr<void> Get(size_t n_bytes) override;
  size_t ReadAtGPU(void *gpu_buffer, size_t n_bytes,
                   ptrdiff_t buffer_offset, int64 file_offset) override;
  size_t ReadGPU(void *buffer, size_t n_bytes, ptrdiff_t offset = 0) override;
  size_t Read(void *cpu_buffer, size_t n_bytes) override;
  void SeekRead(ptrdiff_t pos, int whence = SEEK_SET) override;
  ptrdiff_t TellRead() const override;
  void HandleIOError(int64 ret) const;
  size_t Size() const override;

  ~StdCUFileStream() override;

 private:
  StdCUFileStream(const StdCUFileStream &) = delete;
  StdCUFileStream(StdCUFileStream &&) = delete;
  StdCUFileStream &operator=(const StdCUFileStream &) = delete;
  StdCUFileStream &operator=(StdCUFileStream &&) = delete;
  cufile::CUFileHandle f_ = {};
  size_t length_ = 0;
  size_t pos_ = 0;
};

}  // namespace dali

#endif  // DALI_UTIL_STD_CUFILE_H_
