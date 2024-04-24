// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_S3_FILE_H_
#define DALI_UTIL_S3_FILE_H_

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include "dali/util/file.h"
#include "dali/util/s3_filesystem.h"
#include "dali/util/uri.h"

namespace dali {

class S3FileStream : public FileStream {
 public:
  explicit S3FileStream(Aws::S3::S3Client* s3_client, const std::string& uri,
                        std::optional<size_t> size = std::nullopt);
  void Close() override;
  size_t Read(void* buf, size_t n) override;
  void SeekRead(ptrdiff_t pos, int whence = SEEK_SET) override;
  ptrdiff_t TellRead() const override;
  size_t Size() const override;

  ~S3FileStream() override;

 private:
  Aws::S3::S3Client* s3_client_ = nullptr;
  ptrdiff_t pos_ = 0;
  s3_filesystem::S3ObjectLocation object_location_ = {};
  s3_filesystem::S3ObjectStats object_stats_ = {};
};

}  // namespace dali

#endif  // DALI_UTIL_S3_FILE_H_
