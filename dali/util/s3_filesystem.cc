// Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include "dali/util/s3_filesystem.h"
#include <aws/core/Aws.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/nvtx.h"
#include "dali/util/uri.h"

namespace dali {

namespace s3_filesystem {

static const char kAllocationTag[] = "s3_filesystem";

S3ObjectLocation parse_uri(const std::string& uri) {
  auto parsed_uri = URI::Parse(uri, URI::ParseOpts::AllowNonEscaped);
  if (parsed_uri.scheme() != "s3")
    throw std::runtime_error("Not an S3 URI: " + uri);
  S3ObjectLocation object_location;
  object_location.bucket = parsed_uri.authority();
  object_location.object = parsed_uri.path();
  if (object_location.object.length() >= 1 && object_location.object[0] == '/')
    object_location.object = object_location.object.substr(1);
  return object_location;
}

S3ObjectStats get_stats(Aws::S3::S3Client* s3_client, const S3ObjectLocation& object_location) {
  DomainTimeRange tr(make_string("get_stats @ ", object_location.object), DomainTimeRange::kOrange);
  S3ObjectStats stats;
  if (object_location.object.empty())
    throw std::runtime_error("Object can't be empty");

  Aws::S3::Model::HeadObjectRequest head_object_req;
  head_object_req.SetBucket(object_location.bucket.c_str());
  head_object_req.SetKey(object_location.object.c_str());
  head_object_req.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kAllocationTag); });
  auto head_object_outcome = s3_client->HeadObject(head_object_req);
  if (!head_object_outcome.IsSuccess()) {
    const Aws::S3::S3Error& err = head_object_outcome.GetError();
    throw std::runtime_error("S3 Object not found. bucket=" + object_location.bucket +
                             " object=" + object_location.object + ":\n" + err.GetExceptionName() +
                             ": " + err.GetMessage());
  }
  stats.exists = true;
  stats.size = stats.exists ? head_object_outcome.GetResult().GetContentLength() : 0;
  return stats;
}


size_t read_object_contents(Aws::S3::S3Client* s3_client, const S3ObjectLocation& object_location,
                            void* buf, size_t n, size_t offset) {
  std::stringstream ss;
  ss << "bytes=" << offset << "-" << offset + n - 1;
  std::string byte_range_str = ss.str();

  DomainTimeRange tr(make_string("read_object_contents @ ", object_location.object, " ",
                                 byte_range_str, " (", n, ")"),
                     DomainTimeRange::kOrange);

  Aws::S3::Model::GetObjectRequest getObjectRequest;
  getObjectRequest.SetBucket(object_location.bucket.c_str());
  getObjectRequest.SetKey(object_location.object.c_str());
  getObjectRequest.SetRange(byte_range_str.c_str());

  Aws::Utils::Stream::PreallocatedStreamBuf streambuf(reinterpret_cast<uint8_t*>(buf), n);
  getObjectRequest.SetResponseStreamFactory(
      [&streambuf]() { return Aws::New<Aws::IOStream>(kAllocationTag, &streambuf); });

  size_t bytes_read = 0;
  auto get_object_outcome = s3_client->GetObject(getObjectRequest);
  if (!get_object_outcome.IsSuccess()) {
    const Aws::S3::S3Error& err = get_object_outcome.GetError();
    throw std::runtime_error(err.GetExceptionName() + ": " + err.GetMessage());
  } else {
    bytes_read = get_object_outcome.GetResult().GetContentLength();
  }
  return bytes_read;
}

void list_objects_f(Aws::S3::S3Client* s3_client, const S3ObjectLocation& object_location,
                    PerObjectCallable per_object_call) {
  DomainTimeRange tr(make_string("list_object_contents @ ", object_location.object),
                     DomainTimeRange::kOrange);
  std::string prefix = object_location.object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }
  constexpr int kS3GetChildrenMaxKeys = 1000;
  Aws::S3::Model::ListObjectsV2Request list_obj_req;
  list_obj_req.WithBucket(object_location.bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys);
  list_obj_req.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kAllocationTag); });
  Aws::S3::Model::ListObjectsV2Result list_obj_result;
  std::vector<std::string> results;
  do {
    auto list_obj_outcome = s3_client->ListObjectsV2(list_obj_req);
    if (!list_obj_outcome.IsSuccess()) {
      throw std::runtime_error(list_obj_outcome.GetError().GetExceptionName() + ": " +
                               list_obj_outcome.GetError().GetMessage());
    }
    list_obj_result = list_obj_outcome.GetResult();
    for (const auto& object : list_obj_result.GetContents()) {
      per_object_call(object.GetKey(), object.GetSize());
    }
    list_obj_req.SetContinuationToken(list_obj_result.GetNextContinuationToken());
  } while (list_obj_result.GetIsTruncated());
}

}  // namespace s3_filesystem

}  // namespace dali
