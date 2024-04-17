// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/filesystem.h"
#include <gtest/gtest.h>

namespace dali {

TEST(JoinPath, File) {
  EXPECT_EQ("/path/dir/path2",
            filesystem::join_path("/path/dir", "path2"));
  EXPECT_EQ("/path/dir/path2",
            filesystem::join_path("/path/dir/", "path2"));
  EXPECT_EQ("/path2",
            filesystem::join_path("/path/dir", "/path2"));
}

TEST(JoinPath, URI) {
  EXPECT_EQ("s3://my_bucket/mypath/path2",
            filesystem::join_path("s3://my_bucket/mypath", "path2"));
  EXPECT_EQ("s3://my_bucket/mypath/path2",
            filesystem::join_path("s3://my_bucket/mypath/", "path2"));
  EXPECT_EQ("s3://my_bucket/path2",
            filesystem::join_path("s3://my_bucket/mypath", "/path2"));
  EXPECT_EQ("s3://my_bucket/path2",
            filesystem::join_path("s3://my_bucket/mypath/", "/path2"));
}

}  // namespace dali
