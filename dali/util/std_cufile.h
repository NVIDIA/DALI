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
  size_t ReadGPUImpl(uint8_t* gpu_buffer, size_t n_bytes,
                     size_t buffer_offset, size_t file_offset) override;
  size_t ReadGPU(uint8_t * buffer, size_t n_bytes, size_t offset = 0) override;
  size_t Read(uint8_t* cpu_buffer, size_t n_bytes) override;
  size_t Pos() const;
  void Seek(int64 pos) override;
  int64 Tell() const override;
  void HandleIOError(int64 ret) const;
  size_t Size() const override;

  ~StdCUFileStream() override {
    Close();
  }

 private:
  cufile::CUFileHandle f_ = {};
  size_t length_ = 0;
  size_t pos_ = 0;
};

}  // namespace dali

#endif  // DALI_UTIL_STD_CUFILE_H_
