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

#include "dali/operators/reader/loader/tar_utils.h"
#include <fcntl.h>
#include <gtest/gtest.h>
#include <libtar.h>
#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/util/std_file.h"


namespace dali {
namespace detail {

void TestArchiveEntries(TarArchive& archive, const std::vector<std::string>& prefixes, int beg,
                        int end, bool preread) {
  for (int idx = beg; idx < end; idx++) {
    for (int prefix_idx = 0; prefix_idx < static_cast<int>(prefixes.size()); prefix_idx++) {
      if (preread) {
        archive.Read();
      }
      ASSERT_EQ(archive.GetFileName(), to_string(idx) + prefixes[prefix_idx]);
      bool tmp = archive.NextFile() ^
                 (idx == end - 1 && prefix_idx == static_cast<int>(prefixes.size()) - 1);
      ASSERT_TRUE(tmp);
    }
  }

  ASSERT_FALSE(archive.NextFile());
  ASSERT_EQ(archive.GetFileName(), "");
  ASSERT_EQ(archive.GetFileSize(), 0);
}

struct SimpleTarTestsData {
  std::string filepath;
  bool read_ahead;
  bool use_mmap;
  int beg;
  int end;
  std::vector<std::string> prefixes;
};

class SimpleTarTests : public testing::TestWithParam<SimpleTarTestsData> {};

TEST_P(SimpleTarTests, Index) {
  TarArchive archive(
      FileStream::Open(GetParam().filepath, GetParam().read_ahead, GetParam().use_mmap));
  TestArchiveEntries(archive, GetParam().prefixes, GetParam().beg, GetParam().end, false);
}

TEST_P(SimpleTarTests, PostIndex) {
  TarArchive archive(
      FileStream::Open(GetParam().filepath, GetParam().read_ahead, GetParam().use_mmap));
  TestArchiveEntries(archive, GetParam().prefixes, GetParam().beg, GetParam().end, true);
}

TEST_P(SimpleTarTests, Contents) {
  TAR* handle;
  ASSERT_EQ(tar_open(&handle, GetParam().filepath.c_str(), NULL, O_RDONLY, 0, TAR_GNU), 0);
  TarArchive archive(
      FileStream::Open(GetParam().filepath, GetParam().read_ahead, GetParam().use_mmap));

  do {
    th_read(handle);
    vector<uint8_t> libtar_contents;
    uint64 count = 0;
    uint64 filesize = th_get_size(handle);

    do {
      libtar_contents.resize(libtar_contents.size() + T_BLOCKSIZE);
      tar_block_read(handle, libtar_contents.data() + libtar_contents.size() - T_BLOCKSIZE);
    } while ((count += T_BLOCKSIZE) < filesize);
    libtar_contents.resize(filesize);

    ASSERT_EQ(archive.Read(), libtar_contents);
  } while (archive.NextFile());
  ASSERT_EQ(th_read(handle), 1);
  ASSERT_EQ(tar_close(handle), 0);
}

auto SimpleTarTestsValues() {
  vector<SimpleTarTestsData> values;

  SimpleTarTestsData filepaths[] = {
      {dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                   "db/webdataset/MNIST/devel-0.tar"),
       false,
       false,
       2000,
       3000,
       {".cls", ".jpg"}},
      {dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                   "db/webdataset/sample-tar/empty.tar"),
       false,
       false,
       0,
       0,
       {}},
      {dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                   "db/webdataset/sample-tar/v7.tar"),
       false,
       false,
       0,
       1000,
       {""}},
      {dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                   "db/webdataset/sample-tar/oldgnu.tar"),
       false,
       false,
       0,
       1000,
       {""}}};

  for (auto& filepath : filepaths) {
    for (int read_ahead = 0; read_ahead <= 1; read_ahead++) {
      for (int use_mmap = 0; use_mmap <= 1; use_mmap++) {
        filepath.read_ahead = read_ahead;
        filepath.use_mmap = use_mmap;
        values.push_back(filepath);
      }
    }
  }
  return testing::ValuesIn(values.begin(), values.end());
}

INSTANTIATE_TEST_CASE_P(LibTarUtilsTest, SimpleTarTests, SimpleTarTestsValues());

TEST(LibTarUtilsTest, Interface) {
  std::string filepath(dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                                   "db/webdataset/MNIST/devel-1.tar"));
  TarArchive archive(TarArchive(TarArchive(FileStream::Open(filepath, false, false))));
  ASSERT_TRUE(archive.NextFile());
  ASSERT_TRUE(archive.IsAtFile());
  ASSERT_EQ(archive.GetFileName(), "0.jpg");
  int filesize = archive.GetFileSize();
  ASSERT_FALSE(archive.Eof());
  archive.Read(10);
  ASSERT_FALSE(archive.Eof());
  ASSERT_EQ(archive.GetFileSize(), filesize);
  archive.Read();
  ASSERT_TRUE(archive.Eof());
}

TEST(LibTarUtilsTest, LongNameIndexing) {
  std::string filepath(dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"),
                                                   "db/webdataset/sample-tar/gnu.tar"));
  TarArchive archive(TarArchive(TarArchive(FileStream::Open(filepath, false, false))));
  std::string name_prefix(128, '#');
  for (int idx = 0; idx < 1000; idx++) {
    ASSERT_EQ(archive.GetFileName(), name_prefix + to_string(idx));
    archive.NextFile();
  }
}

TEST(LibTarUtilsTest, MultithreadedIndex) {
  std::string filepath_prefix(
      dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"), "db/webdataset/MNIST/devel-"));

  std::string filepaths[3] = {filepath_prefix + "0.tar", filepath_prefix + "1.tar",
                              filepath_prefix + "2.tar"};

  std::unique_ptr<TarArchive> archives[3] = {
      std::make_unique<TarArchive>(FileStream::Open(filepaths[0], false, false)),
      std::make_unique<TarArchive>(FileStream::Open(filepaths[1], false, false)),
      std::make_unique<TarArchive>(FileStream::Open(filepaths[2], false, false))};

  std::future<void> threads[3] = {
      std::async(std::launch::async,
                 [&] {
                   TestArchiveEntries(*archives[0], {".cls", ".jpg"}, 2000, 3000, false);
                 }),
      std::async(std::launch::async,
                 [&] {
                   TestArchiveEntries(*archives[1], {".cls", ".jpg"}, 0, 1000, false);
                 }),
      std::async(std::launch::async, [&] {
        TestArchiveEntries(*archives[2], {".cls", ".jpg"}, 1000, 2000, false);
      })};

  std::for_each(threads, threads + 3, [](auto& thread) { thread.wait(); });
}

TEST(LibTarUtilsTest, MultithreadedPostIndex) {
  std::string filepath_prefix(
      dali::filesystem::join_path(std::getenv("DALI_EXTRA_PATH"), "db/webdataset/MNIST/devel-"));

  std::string filepaths[3] = {filepath_prefix + "0.tar", filepath_prefix + "1.tar",
                              filepath_prefix + "2.tar"};

  std::unique_ptr<TarArchive> archives[3] = {
      std::make_unique<TarArchive>(FileStream::Open(filepaths[0], false, false)),
      std::make_unique<TarArchive>(FileStream::Open(filepaths[1], false, false)),
      std::make_unique<TarArchive>(FileStream::Open(filepaths[2], false, false))};

  std::future<void> threads[3] = {
      std::async(std::launch::async,
                 [&] {
                   TestArchiveEntries(*archives[0], {".cls", ".jpg"}, 2000, 3000, true);
                 }),
      std::async(std::launch::async,
                 [&] {
                   TestArchiveEntries(*archives[1], {".cls", ".jpg"}, 0, 1000, true);
                 }),
      std::async(std::launch::async, [&] {
        TestArchiveEntries(*archives[2], {".cls", ".jpg"}, 1000, 2000, true);
      })};

  std::for_each(threads, threads + 3, [](auto& thread) { thread.wait(); });
}


}  // namespace detail
}  // namespace dali
