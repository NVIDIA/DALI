// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include <fcntl.h>
#include <gtest/gtest.h>
#include <libtar.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/util.h"
#include "dali/operators/reader/loader/discover_files.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/file.h"

namespace dali {
namespace detail {

TEST(LibTarUtilsTest, ReadsPayloadWithoutBlockPadding) {
  class MemoryFileStream : public FileStream {
   public:
    explicit MemoryFileStream(const std::string &data)
        : FileStream("memory.tar"), stream_(data.data(), data.size()) {}
    size_t Read(void *buf, size_t n) override { return stream_.Read(buf, n); }
    void SeekRead(ptrdiff_t pos, int whence) override { stream_.SeekRead(pos, whence); }
    ssize_t TellRead() const override { return stream_.TellRead(); }
    size_t Size() const override { return stream_.Size(); }
    void Close() override {}

   private:
    MemInputStream stream_;
  };

  tar_header header = {};
  std::memcpy(header.name, "sample.bin", 10);
  std::memcpy(header.magic, "ustar", 5);
  std::memcpy(header.version, "00", 2);
  header.typeflag = REGTYPE;
  std::snprintf(header.size, sizeof(header.size), "%011o", 4);
  std::memset(header.chksum, ' ', sizeof(header.chksum));
  unsigned checksum = 0;
  auto *raw_header = reinterpret_cast<const unsigned char *>(&header);
  for (int i = 0; i < T_BLOCKSIZE; i++)
    checksum += raw_header[i];
  std::snprintf(header.chksum, sizeof(header.chksum), "%06o", checksum);
  header.chksum[7] = ' ';
  std::string data(reinterpret_cast<const char *>(&header), T_BLOCKSIZE);
  data += "abcd";

  TarArchive archive(std::make_unique<MemoryFileStream>(data));
  ASSERT_EQ(archive.GetFileSize(), 4u);
  char actual[4];
  ASSERT_EQ(archive.Read(actual, sizeof(actual)), sizeof(actual));
  EXPECT_EQ(std::string(actual, sizeof(actual)), "abcd");
}

namespace {

// Builds a 12-byte tar size field the way writers do: right-justified in 11 characters + NUL.
template <typename Field>
void SetOctalField(Field &field, const char *digits) {
  std::memset(field, 0, sizeof(field));
  size_t n = std::strlen(digits);
  std::memset(field, ' ', 11);
  std::memcpy(field + 11 - n, digits, n);
}

}  // namespace

TEST(TarSizeParsing, Octal) {
  char field[12];
  SetOctalField(field, "4000");
  EXPECT_EQ(ParseTarSize(field), 2048u);
  SetOctalField(field, "0");
  EXPECT_EQ(ParseTarSize(field), 0u);
  // The largest value a 11-digit field can hold needs more than 32 bits.
  SetOctalField(field, "77777777777");
  EXPECT_EQ(ParseTarSize(field), 8589934591u);
  // 4 GiB + 512: a 32-bit parser (stock libtar) would report 512 here.
  SetOctalField(field, "40000001000");
  EXPECT_EQ(ParseTarSize(field), 4294967808u);
}

TEST(TarSizeParsing, FullFieldWithoutTerminatorIsBounded) {
  // 12 octal digits and no NUL: the parser must not read into the neighbouring header field.
  struct {
    char size[12];
    char mtime[12];
  } fields;
  std::memset(fields.size, '7', sizeof(fields.size));
  std::memset(fields.mtime, '7', sizeof(fields.mtime));
  EXPECT_EQ(ParseTarSize(fields.size), (uint64_t(1) << 36) - 1);
}

TEST(TarSizeParsing, MalformedFields) {
  char field[12];
  // Same convention as libtar's oct_to_size: no digits -> 0.
  std::memset(field, 0, sizeof(field));
  EXPECT_EQ(ParseTarSize(field), 0u);
  std::memcpy(field, "garbage\0\0\0\0\0", 12);
  EXPECT_EQ(ParseTarSize(field), 0u);
  // A negative number wraps modulo 2^64, like the 64-bit libtar fork does.
  SetOctalField(field, "-4");
  EXPECT_EQ(ParseTarSize(field), uint64_t(0) - 4);
}

TEST(TarSizeParsing, Base256) {
  unsigned char field[12] = {0x80, 0, 0, 0, 0, 0, 0, 0x01, 0x40, 0, 0, 0};  // 5 GiB
  EXPECT_EQ(ParseTarSize(reinterpret_cast<const char (&)[12]>(field)), uint64_t(5) << 30);
  // Negative (0xFF prefix) and values that do not fit in 64 bits are not representable;
  // they must come out as something no archive can satisfy.
  unsigned char negative[12] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                0xFC};
  EXPECT_EQ(ParseTarSize(reinterpret_cast<const char (&)[12]>(negative)),
            std::numeric_limits<uint64_t>::max());
  unsigned char too_big[12] = {0x80, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(ParseTarSize(reinterpret_cast<const char (&)[12]>(too_big)),
            std::numeric_limits<uint64_t>::max());
}

TEST(LibTarUtilsTestSimple, Interface) {
  std::string filepath(dali::filesystem::join_path(testing::dali_extra_path(),
                                                   "db/webdataset/MNIST/devel-2.tar"));
  std::string dummy_filepath(dali::filesystem::join_path(testing::dali_extra_path(),
                                                         "db/webdataset/MNIST/devel-1.tar"));

  TarArchive dummy_archive(FileStream::Open(dummy_filepath));
  TarArchive intermediate_archive(std::move(dummy_archive));
  TarArchive archive(FileStream::Open(filepath));
  archive = std::move(intermediate_archive);

  ASSERT_FALSE(dummy_archive.NextFile());
  ASSERT_TRUE(dummy_archive.EndOfArchive());
  ASSERT_FALSE(intermediate_archive.NextFile());
  ASSERT_TRUE(intermediate_archive.EndOfArchive());

  ASSERT_TRUE(archive.NextFile());
  ASSERT_FALSE(archive.EndOfArchive());

  ASSERT_EQ(archive.GetFileName(), "0.jpg");
  ASSERT_EQ(archive.GetFileType(), TarArchive::ENTRY_FILE);
  int filesize = archive.GetFileSize();
  ASSERT_FALSE(archive.EndOfFile());

  vector<uint8_t> buffer(filesize);
  archive.Read(buffer.data(), 10_u8);
  ASSERT_FALSE(archive.EndOfFile());
  ASSERT_EQ(archive.GetFileSize(), filesize);

  ASSERT_TRUE(archive.ReadFile() == nullptr);

  archive.Read(buffer.data() + 10, filesize);
  ASSERT_TRUE(archive.EndOfFile());
  ASSERT_EQ(archive.GetFileSize(), filesize);
}

TEST(LibTarUtilsTestSimple, LongNameIndexing) {
  std::string filepath(dali::filesystem::join_path(testing::dali_extra_path(),
                                                   "db/webdataset/sample-tar/gnu.tar"));
  TarArchive archive(FileStream::Open(filepath));
  std::string name_prefix(128, '#');
  for (int idx = 0; idx < 1000; idx++) {
    ASSERT_EQ(archive.GetFileName(), name_prefix + to_string(idx));
    ASSERT_EQ(archive.GetFileType(), TarArchive::ENTRY_FILE);
    archive.NextFile();
  }
}

TEST(LibTarUtilsTestSimple, Types) {
  std::string filepath(dali::filesystem::join_path(testing::dali_extra_path(),
                                                   "db/webdataset/sample-tar/types.tar"));
  std::vector<TarArchive::EntryType> types = {
      TarArchive::ENTRY_BLOCKDEV, TarArchive::ENTRY_CHARDEV,  TarArchive::ENTRY_DIR,
      TarArchive::ENTRY_FIFO,     TarArchive::ENTRY_FILE,     TarArchive::ENTRY_SYMLINK,
      TarArchive::ENTRY_HARDLINK, TarArchive::ENTRY_BLOCKDEV, TarArchive::ENTRY_CHARDEV,
      TarArchive::ENTRY_DIR,      TarArchive::ENTRY_FIFO,     TarArchive::ENTRY_FILE,
      TarArchive::ENTRY_SYMLINK,  TarArchive::ENTRY_HARDLINK};

  FileStream::Options opts;
  opts.read_ahead = false;
  opts.use_mmap = true;
  opts.use_odirect = false;
  TarArchive archive(FileStream::Open(filepath, opts));
  for (size_t i = 0; i < types.size(); i++) {
    ASSERT_EQ(archive.GetFileType(), types[i]);
    ASSERT_EQ(archive.GetFileName(), to_string(i) + (types[i] == TarArchive::ENTRY_DIR ? "/" : ""));
    ASSERT_EQ(archive.GetFileSize(), 0_uz);
    ASSERT_EQ(archive.TellArchive(), i * T_BLOCKSIZE);
    archive.NextFile();
  }
  ASSERT_TRUE(archive.EndOfArchive());
}

TEST(LibTarUtilsTestSimple, Offset) {
  std::string filepath(dali::filesystem::join_path(testing::dali_extra_path(),
                                                   "db/webdataset/sample-tar/types.tar"));

  FileStream::Options opts;
  opts.read_ahead = false;
  opts.use_mmap = true;
  opts.use_odirect = false;
  TarArchive archive(FileStream::Open(filepath, opts));
  archive.SeekArchive(7 * T_BLOCKSIZE);
  ASSERT_EQ(archive.TellArchive(), 7 * T_BLOCKSIZE);
  for (int i = 7; i < 14; i++) {
    ASSERT_EQ(archive.GetFileName(),
              to_string(i) + (archive.GetFileType() == TarArchive::ENTRY_DIR ? "/" : ""));
    archive.NextFile();
  }
  ASSERT_TRUE(archive.EndOfArchive());
}

void TestArchiveEntries(TarArchive& archive, const std::vector<std::string>& prefixes, int beg,
                        int end, bool preread) {
  size_t total_size = 0;
  for (int idx = beg; idx < end; idx++) {
    for (size_t prefix_idx = 0; prefix_idx < prefixes.size(); prefix_idx++) {
      if (preread) {
        archive.ReadFile();
      }
      ASSERT_EQ(archive.GetFileName(), to_string(idx) + prefixes[prefix_idx]);
      ASSERT_EQ(archive.GetFileType(), TarArchive::ENTRY_FILE);
      ASSERT_EQ(archive.TellArchive(), total_size);
      total_size += align_up(archive.GetFileSize(), T_BLOCKSIZE) + T_BLOCKSIZE;
      ASSERT_TRUE(archive.NextFile() ^ (idx == end - 1 && prefix_idx == prefixes.size() - 1));
    }
  }

  ASSERT_FALSE(archive.NextFile());
  ASSERT_EQ(archive.GetFileName(), "");
  ASSERT_EQ(archive.GetFileType(), TarArchive::ENTRY_NONE);
  ASSERT_EQ(archive.GetFileSize(), 0_uz);
}

struct SimpleTarTestsData {
  std::string filepath;
  bool read_ahead;
  bool use_mmap;
  int beg;
  int end;
  std::vector<std::string> prefixes;
};

class SimpleTarTests : public ::testing::TestWithParam<SimpleTarTestsData> {
 protected:
  TarArchive archive;
  SimpleTarTests()
      : archive(FileStream::Open(GetParam().filepath,
                                 {GetParam().read_ahead, GetParam().use_mmap, false})) {}
};

TEST_P(SimpleTarTests, Index) {
  TestArchiveEntries(archive, GetParam().prefixes, GetParam().beg, GetParam().end, false);
}

TEST_P(SimpleTarTests, PostIndex) {
  TestArchiveEntries(archive, GetParam().prefixes, GetParam().beg, GetParam().end, true);
}

TEST_P(SimpleTarTests, Contents) {
  TAR* handle;
  ASSERT_EQ(tar_open(&handle, GetParam().filepath.c_str(), NULL, O_RDONLY, 0, TAR_GNU), 0);

  while (!archive.EndOfArchive()) {
    ASSERT_EQ(th_read(handle), 0);
    vector<uint8_t> libtar_contents;
    uint64_t count = 0;
    uint64_t filesize = th_get_size(handle);
    ASSERT_EQ(archive.GetFileSize(), filesize);

    do {
      libtar_contents.resize(libtar_contents.size() + T_BLOCKSIZE);
      ASSERT_EQ(
          tar_block_read(handle, libtar_contents.data() + libtar_contents.size() - T_BLOCKSIZE),
          T_BLOCKSIZE);
    } while ((count += T_BLOCKSIZE) < filesize);
    libtar_contents.resize(filesize);

    vector<uint8_t> file(archive.GetFileSize());
    archive.Read(file.data(), archive.GetFileSize());

    for (size_t idx = 0; idx < libtar_contents.size(); idx++) {
      ASSERT_EQ(file[idx], libtar_contents[idx]);
    }
    archive.NextFile();
  }
  ASSERT_EQ(th_read(handle), 1);
  ASSERT_EQ(tar_close(handle), 0);
}

auto SimpleTarTestsValues() {
  vector<SimpleTarTestsData> values;

  SimpleTarTestsData filepaths[] = {
      { dali::filesystem::join_path(testing::dali_extra_path(),
                                    "db/webdataset/MNIST/devel-0.tar"),
        false,
        false,
        2000,
        3000,
        {".cls", ".jpg"} },
      { dali::filesystem::join_path(testing::dali_extra_path(),
                                    "db/webdataset/sample-tar/empty.tar"),
        false,
        false,
        0,
        0,
        {} },
      { dali::filesystem::join_path(testing::dali_extra_path(),
                                    "db/webdataset/sample-tar/v7.tar"),
        false,
        false,
        0,
        1000,
        {""} },
      { dali::filesystem::join_path(testing::dali_extra_path(),
                                    "db/webdataset/sample-tar/oldgnu.tar"),
        false,
        false,
        0,
        1000,
        {""} } };

  for (auto& filepath : filepaths) {
    for (int read_ahead = 0; read_ahead <= 1; read_ahead++) {
      for (int use_mmap = 0; use_mmap <= 1; use_mmap++) {
        filepath.read_ahead = read_ahead;
        filepath.use_mmap = use_mmap;
        values.push_back(filepath);
      }
    }
  }
  return ::testing::ValuesIn(values.begin(), values.end());
}

INSTANTIATE_TEST_SUITE_P(LibTarUtilsTestParametrized, SimpleTarTests, SimpleTarTestsValues());

constexpr int kMultithreadedSamples = 3;

class MultiTarTests : public ::testing::TestWithParam<bool> {
 protected:
  std::unique_ptr<TarArchive> archives[kMultithreadedSamples];
  const std::pair<int, int> ranges[kMultithreadedSamples] = {{2000, 3000}, {0, 1000}, {1000, 2000}};

  void SetUp() final {
    std::string filepath_prefix(
        dali::filesystem::join_path(testing::dali_extra_path(), "db/webdataset/MNIST/devel-"));

    std::string filepaths[kMultithreadedSamples] = {
        filepath_prefix + "0.tar", filepath_prefix + "1.tar", filepath_prefix + "2.tar"};

    for (int idx = 0; idx < kMultithreadedSamples; idx++) {
      archives[idx] = std::make_unique<TarArchive>(FileStream::Open(filepaths[idx]));
    }
  }
};

TEST_P(MultiTarTests, Index) {
  std::future<void> tasks[kMultithreadedSamples];

  for (int idx = 0; idx < kMultithreadedSamples; idx++) {
    tasks[idx] = std::async(std::launch::async, [=, this] {
      TestArchiveEntries(*archives[idx], {".cls", ".jpg"}, ranges[idx].first, ranges[idx].second,
                         GetParam());
    });
  }

  for (auto& task : tasks) {
    task.wait();
  }
}

INSTANTIATE_TEST_SUITE_P(LibTarUtilsTestMultithreaded, MultiTarTests,
                        ::testing::Values(false, true));

}  // namespace detail
}  // namespace dali
