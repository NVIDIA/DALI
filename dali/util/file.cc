// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "dali/util/file.h"
#include "dali/util/mmaped_file.h"
#include "dali/util/odirect_file.h"
#include "dali/util/std_file.h"
#include "dali/util/uri.h"

#if AWSSDK_ENABLED
#include "dali/util/s3_client_manager.h"
#include "dali/util/s3_file.h"
#endif

namespace dali {

std::unique_ptr<FileStream> FileStream::Open(const std::string& uri, FileStream::Options opts,
                                             std::optional<size_t> size) {
  bool is_s3 = uri.rfind("s3://", 0) == 0;
  if (is_s3) {
#if AWSSDK_ENABLED
    return std::make_unique<S3FileStream>(S3ClientManager::Instance().client(), uri, size);
#else
    throw std::runtime_error("This version of DALI was not built with AWS S3 storage support.");
#endif
  }

  std::string processed_uri;
  if (uri.find("file://") == 0) {
    processed_uri = uri.substr(std::string("file://").size());
  } else {
    processed_uri = uri;
  }

  if (opts.use_mmap) {
    return std::unique_ptr<FileStream>(new MmapedFileStream(processed_uri, opts.read_ahead));
  } else if (opts.use_odirect) {
    return std::unique_ptr<FileStream>(new ODirectFileStream(processed_uri));
  } else {
    return std::unique_ptr<FileStream>(new StdFileStream(processed_uri));
  }
}

bool FileStream::ReserveFileMappings(unsigned int num) {
  return MmapedFileStream::ReserveFileMappings(num);
}
void FileStream::FreeFileMappings(unsigned int num) {
  MmapedFileStream::FreeFileMappings(num);
}

}  // namespace dali
