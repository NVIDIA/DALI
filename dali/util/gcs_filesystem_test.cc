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
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace dali {
namespace gcs_filesystem {

TEST(GCSParseURI, BucketAndObject) {
  auto loc = parse_uri("gs://my-bucket/path/to/object.dat");
  EXPECT_EQ("my-bucket", loc.bucket);
  EXPECT_EQ("path/to/object.dat", loc.object);
}

TEST(GCSParseURI, LeadingSlashIsStripped) {
  // URI::Parse keeps the leading '/' in the path; the object name must not have it, or every
  // request would address an object whose name starts with a slash.
  auto loc = parse_uri("gs://my-bucket/object.dat");
  EXPECT_EQ("object.dat", loc.object);
}

TEST(GCSParseURI, BucketOnly) {
  auto loc = parse_uri("gs://my-bucket");
  EXPECT_EQ("my-bucket", loc.bucket);
  EXPECT_EQ("", loc.object);
}

TEST(GCSParseURI, BucketWithTrailingSlash) {
  auto loc = parse_uri("gs://my-bucket/");
  EXPECT_EQ("my-bucket", loc.bucket);
  EXPECT_EQ("", loc.object);
}

TEST(GCSParseURI, TrailingSlashIsKept) {
  // A name ending with '/' is how GCS spells a directory marker - it must survive parsing, and
  // it is also what a prefix looks like when the user passes one.
  auto loc = parse_uri("gs://my-bucket/data/class_a/");
  EXPECT_EQ("data/class_a/", loc.object);
}

TEST(GCSParseURI, UnescapedCharactersAreAllowed) {
  auto loc = parse_uri("gs://my-bucket/data/class_a/0 x.dat");
  EXPECT_EQ("data/class_a/0 x.dat", loc.object);
}

TEST(GCSParseURI, RejectsOtherSchemes) {
  EXPECT_THROW(parse_uri("s3://my-bucket/object.dat"), std::runtime_error);
  EXPECT_THROW(parse_uri("file:///tmp/object.dat"), std::runtime_error);
  EXPECT_THROW(parse_uri("/tmp/object.dat"), std::runtime_error);
}

}  // namespace gcs_filesystem
}  // namespace dali
