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

#ifndef DALI_UTIL_S3_FILESYSTEM_H_
#define DALI_UTIL_S3_FILESYSTEM_H_

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <cstdio>
#include <functional>
#include <string>
#include "dali/core/api_helper.h"

namespace dali {

namespace s3_filesystem {

struct DLL_PUBLIC S3ObjectLocation {
  std::string bucket;
  std::string object;
};

struct DLL_PUBLIC S3ObjectStats {
  bool exists = false;
  size_t size = 0;
};

/**
 * @brief Parses an S3 URI into an object location
 *
 * @param uri URI to the S3 prefix to query
 * @return S3ObjectLocation object location
 */
DLL_PUBLIC S3ObjectLocation parse_uri(const std::string& uri);

/**
 * @brief Get the s3 object or bucket stats
 *
 * @param s3_client open S3 client
 * @param object_location S3 object location
 * @return S3ObjectStats object stats
 */
DLL_PUBLIC S3ObjectStats get_stats(Aws::S3::S3Client* s3_client,
                                   const S3ObjectLocation& object_location);

/**
 * @brief Read S3 object contents
 *
 * @param s3_client open S3 client
 * @param object_location object location
 * @param buf preallocated buffer location
 * @param n number of bytes to read
 * @param offset (optional) offset to start reading from
 * @return size_t number of bytes read
 */
DLL_PUBLIC size_t read_object_contents(Aws::S3::S3Client* s3_client,
                                       const S3ObjectLocation& object_location, void* buf, size_t n,
                                       size_t offset = 0);

using PerObjectCallable = std::function<void(const std::string&, size_t)>;

/**
 * @brief Visits all objects under a given object location
 *
 * @param s3_client open S3 client
 * @param object_location S3 object location
 * @param per_object_call callable to run on each object listed
 */
DLL_PUBLIC void list_objects_f(Aws::S3::S3Client* s3_client,
                               const S3ObjectLocation& object_location,
                               PerObjectCallable per_object_call);

}  // namespace s3_filesystem

}  // namespace dali

#endif  // DALI_UTIL_S3_FILESYSTEM_H_
