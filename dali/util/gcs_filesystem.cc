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

#include "dali/util/gcs_filesystem.h"
#include <google/cloud/status.h>
#include <google/cloud/storage/client.h>
#include <cstdint>
#include <string>
#include "dali/core/format.h"
#include "dali/core/nvtx.h"
#include "dali/util/uri.h"

namespace gcs = ::google::cloud::storage;

namespace dali {

namespace gcs_filesystem {

namespace {

std::string error_message(const google::cloud::Status& status) {
  return make_string("[", google::cloud::StatusCodeToString(status.code()), "] ",
                     status.message());
}

}  // namespace

GCSObjectLocation parse_uri(const std::string& uri) {
  auto parsed_uri = URI::Parse(uri, URI::ParseOpts::AllowNonEscaped);
  if (parsed_uri.scheme() != "gs")
    throw std::runtime_error("Not a GCS URI: " + uri);
  GCSObjectLocation object_location;
  object_location.bucket = parsed_uri.authority();
  object_location.object = parsed_uri.path();
  if (object_location.object.length() >= 1 && object_location.object[0] == '/')
    object_location.object = object_location.object.substr(1);
  return object_location;
}

GCSObjectStats get_stats(gcs::Client& client, const GCSObjectLocation& object_location) {
  DomainTimeRange tr(make_string("get_stats @ ", object_location.object), DomainTimeRange::kOrange);
  GCSObjectStats stats;
  if (object_location.object.empty())
    throw std::runtime_error("Object can't be empty");

  auto metadata = client.GetObjectMetadata(object_location.bucket, object_location.object);
  if (!metadata) {
    throw std::runtime_error("GCS object not found. bucket=" + object_location.bucket +
                             " object=" + object_location.object + ":\n" +
                             error_message(metadata.status()));
  }
  stats.exists = true;
  stats.size = metadata->size();
  return stats;
}

size_t read_object_contents(gcs::Client& client, const GCSObjectLocation& object_location,
                            void* buf, size_t n, size_t offset) {
  if (n == 0)
    return 0;
  // ReadRange is right-open ([begin, end)), unlike the HTTP "Range: bytes=first-last" header,
  // which is inclusive on both ends.
  auto begin = static_cast<std::int64_t>(offset);
  auto end = static_cast<std::int64_t>(offset + n);

  DomainTimeRange tr(make_string("read_object_contents @ ", object_location.object, " [", begin,
                                 ", ", end, ") (", n, ")"),
                     DomainTimeRange::kOrange);

  auto stream = client.ReadObject(object_location.bucket, object_location.object,
                                  gcs::ReadRange(begin, end));
  // Unformatted I/O - the data lands directly in the caller's buffer, no intermediate copy.
  stream.read(static_cast<char*>(buf), n);
  // Reading fewer than n bytes (end of object) sets failbit/eofbit, which is not an error here,
  // so the transfer status is the only thing worth checking.
  auto bytes_read = static_cast<size_t>(stream.gcount());
  stream.Close();
  if (!stream.status().ok()) {
    throw std::runtime_error("Failed to read GCS object. bucket=" + object_location.bucket +
                             " object=" + object_location.object + ":\n" +
                             error_message(stream.status()));
  }
  return bytes_read;
}

void list_objects_f(gcs::Client& client, const GCSObjectLocation& object_location,
                    PerObjectCallable per_object_call) {
  DomainTimeRange tr(make_string("list_object_contents @ ", object_location.object),
                     DomainTimeRange::kOrange);
  std::string prefix = object_location.object;
  if (!prefix.empty() && prefix.back() != '/') {
    prefix.push_back('/');
  }
  // ListObjects returns a lazy range that pages through the results transparently.
  for (auto& metadata : client.ListObjects(object_location.bucket, gcs::Prefix(prefix))) {
    if (!metadata) {
      throw std::runtime_error("Failed to list GCS objects. bucket=" + object_location.bucket +
                               " prefix=" + prefix + ":\n" + error_message(metadata.status()));
    }
    per_object_call(metadata->name(), metadata->size());
  }
}

}  // namespace gcs_filesystem

}  // namespace dali
