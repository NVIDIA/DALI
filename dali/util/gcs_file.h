// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_GCS_FILE_H_
#define DALI_UTIL_GCS_FILE_H_

#include <google/cloud/storage/client.h>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include "dali/util/file.h"
#include "dali/util/gcs_filesystem.h"
#include "dali/util/uri.h"

namespace dali {

class GCSFileStream : public FileStream {
 public:
  /**
   * @param client a client obtained from GCSClientManager. It is held by value, because
   *               `google::cloud::storage::Client` instances must not be shared between threads,
   *               while copies of them are cheap and share the connection pool.
   */
  explicit GCSFileStream(google::cloud::storage::Client client, const std::string& uri,
                         std::optional<size_t> size = std::nullopt);
  void Close() override;
  size_t Read(void* buf, size_t n) override;
  void SeekRead(ptrdiff_t pos, int whence = SEEK_SET) override;
  ptrdiff_t TellRead() const override;
  size_t Size() const override;

  ~GCSFileStream() override;

 private:
  google::cloud::storage::Client client_;
  ptrdiff_t pos_ = 0;
  gcs_filesystem::GCSObjectLocation object_location_ = {};
  gcs_filesystem::GCSObjectStats object_stats_ = {};
};

}  // namespace dali

#endif  // DALI_UTIL_GCS_FILE_H_
