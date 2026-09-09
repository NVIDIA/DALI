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

#ifndef DALI_UTIL_GCS_FILESYSTEM_H_
#define DALI_UTIL_GCS_FILESYSTEM_H_

#include <google/cloud/storage/client.h>
#include <cstdio>
#include <functional>
#include <string>
#include "dali/core/api_helper.h"

namespace dali {

namespace gcs_filesystem {

struct DLL_PUBLIC GCSObjectLocation {
  std::string bucket;
  std::string object;
};

struct DLL_PUBLIC GCSObjectStats {
  bool exists = false;
  size_t size = 0;
};

/**
 * @brief Parses a GCS URI (gs://bucket/object) into an object location
 *
 * @param uri URI to the GCS prefix to query
 * @return GCSObjectLocation object location
 */
DLL_PUBLIC GCSObjectLocation parse_uri(const std::string& uri);

/**
 * @brief Get the GCS object stats
 *
 * @param client GCS client. Note that `google::cloud::storage::Client` must not be used
 *               concurrently from multiple threads, but copies of it are cheap and share the
 *               underlying connection pool - see GCSClientManager.
 * @param object_location GCS object location
 * @return GCSObjectStats object stats
 */
DLL_PUBLIC GCSObjectStats get_stats(google::cloud::storage::Client& client,
                                    const GCSObjectLocation& object_location);

/**
 * @brief Read GCS object contents
 *
 * @param client GCS client (see the note in get_stats)
 * @param object_location object location
 * @param buf preallocated buffer location
 * @param n number of bytes to read
 * @param offset (optional) offset to start reading from
 * @return size_t number of bytes read
 */
DLL_PUBLIC size_t read_object_contents(google::cloud::storage::Client& client,
                                       const GCSObjectLocation& object_location, void* buf,
                                       size_t n, size_t offset = 0);

using PerObjectCallable = std::function<void(const std::string&, size_t)>;

/**
 * @brief Visits all objects under a given object location
 *
 * @param client GCS client (see the note in get_stats)
 * @param object_location GCS object location
 * @param per_object_call callable to run on each object listed
 */
DLL_PUBLIC void list_objects_f(google::cloud::storage::Client& client,
                               const GCSObjectLocation& object_location,
                               PerObjectCallable per_object_call);

}  // namespace gcs_filesystem

}  // namespace dali

#endif  // DALI_UTIL_GCS_FILESYSTEM_H_
