// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <libtar.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <cstdlib>
#include <string>
#include <iostream>

#include "dali/operators/reader/loader/filesystem.h"

namespace dali {
namespace testing {

TEST(LibTar, OpenClose) {
  std::string filepath(dali::filesystem::join_path(
      std::getenv("DALI_EXTRA_PATH"),
      "db/webdataset/MNIST/devel-0.tar"));

  TAR* archive;
  ASSERT_EQ(tar_open(&archive, filepath.c_str(), NULL, O_RDONLY, 0, TAR_GNU), 0);
  ASSERT_EQ(tar_close(archive), 0);
}

}  // namespace testing
}  // namespace dali
