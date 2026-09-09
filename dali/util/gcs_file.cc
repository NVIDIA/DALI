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

#include "dali/util/gcs_file.h"
#include <utility>
#include "dali/core/format.h"
#include "dali/util/gcs_client_manager.h"
#include "dali/util/uri.h"

namespace dali {

GCSFileStream::GCSFileStream(google::cloud::storage::Client client, const std::string& uri,
                             std::optional<size_t> size)
    : FileStream(uri), client_(std::move(client)) {
  object_location_ = gcs_filesystem::parse_uri(uri);
  if (size.has_value() && size.value() > 0) {
    object_stats_.exists = true;
    object_stats_.size = size.value();
  } else {
    object_stats_ = gcs_filesystem::get_stats(client_, object_location_);
  }
}

GCSFileStream::~GCSFileStream() {}

void GCSFileStream::Close() {
  // nothing to do here (there's no file open)
}

void GCSFileStream::SeekRead(ptrdiff_t pos, int whence) {
  auto new_pos = pos_;
  switch (whence) {
    case SEEK_SET:
      new_pos = pos;
      break;
    case SEEK_CUR:
      new_pos += pos;
      break;
    case SEEK_END:
      new_pos = object_stats_.size + pos;
      break;
    default:
      assert(false);
  }
  if (new_pos < 0 || new_pos > static_cast<ptrdiff_t>(object_stats_.size))
    throw std::out_of_range("The requested offset points outside of the file.");
  pos_ = new_pos;
}

ptrdiff_t GCSFileStream::TellRead() const {
  return pos_;
}

size_t GCSFileStream::Size() const {
  return object_stats_.size;
}

size_t GCSFileStream::Read(void* buf, size_t n) {
  if (n == 0)
    return 0;
  size_t bytes_read =
      gcs_filesystem::read_object_contents(client_, object_location_, buf, n, pos_);
  pos_ += bytes_read;
  return bytes_read;
}

}  // namespace dali
