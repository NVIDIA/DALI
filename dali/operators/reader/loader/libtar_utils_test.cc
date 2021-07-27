// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/reader/loader/libtar_utils.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <libtar.h>

#include <string>
#include <utility>

#include "dali/operators/reader/loader/filesystem.h"

namespace dali {
namespace detail {

TEST(LibTarUtilsTest, Index) {
  std::string filepath(dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                                   "db/webdataset/MNIST/devel-0.tar"));

  TarArchive archive(filepath);
  for (int i = 2000; i <= 2999; i++) {
    ASSERT_EQ(archive.GetFileName(), to_string(i) + ".cls");
    ASSERT_TRUE(archive.NextFile());
    ASSERT_EQ(archive.GetFileName(), to_string(i) + ".jpg");
    ASSERT_TRUE(archive.NextFile() ^ (i == 2999));
  }
  ASSERT_FALSE(archive.NextFile());
}

TEST(LibTarUtilsTest, PostReadIndex) {
  std::string filepath(dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                                   "db/webdataset/MNIST/devel-0.tar"));

  TarArchive archive(filepath);
  for (int i = 2000; i <= 2999; i++) {
    archive.Read();
    ASSERT_EQ(archive.GetFileName(), to_string(i) + ".cls");
    ASSERT_TRUE(archive.NextFile());
    archive.Read();
    ASSERT_EQ(archive.GetFileName(), to_string(i) + ".jpg");
    ASSERT_TRUE(archive.NextFile() ^ (i == 2999));
  }
  ASSERT_FALSE(archive.NextFile());
}

TEST(LibTarUtilsTest, Contents) {
  std::string filepath(dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                                   "db/webdataset/MNIST/devel-0.tar"));

  TAR *handle;
  ASSERT_EQ(tar_open(&handle, filepath.c_str(), NULL, O_RDONLY, 0, TAR_GNU), 0);
  TarArchive archive(filepath);

  char buf[T_BLOCKSIZE];
  do {
    th_read(handle);
    std::string libtar_contents = "";
    uint64 count = 0;
    uint64 filesize = th_get_size(handle);

    do {
      tar_block_read(handle, buf);
      libtar_contents.append(buf, std::min(static_cast<uint64>(T_BLOCKSIZE), filesize - count));
    } while ((count += T_BLOCKSIZE) < filesize);

    ASSERT_EQ(archive.Read(), libtar_contents);
  } while (archive.NextFile());
  ASSERT_EQ(th_read(handle), 1);
  ASSERT_EQ(tar_close(handle), 0);
}

}  // namespace detail
}  // namespace dali