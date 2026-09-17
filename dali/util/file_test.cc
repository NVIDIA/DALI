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

#include "dali/util/file.h"
#include <gtest/gtest.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace dali {

namespace {

// A payload big enough to force several underflow() refills at a small buffer size, including a
// final chunk that doesn't land on a buffer-size boundary.
std::string MakePayload(size_t n) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; i++)
    s[i] = static_cast<char>('A' + (i % 26));
  return s;
}

// Models ODirectFileStream: a raw read() syscall that may return fewer bytes than requested
// without that meaning EOF. Every call here artificially returns at most `short_read_cap` bytes,
// even when more of the requested range is available and it isn't the end of the stream.
class ShortReadStream : public FileStream {
 public:
  ShortReadStream(std::string data, size_t short_read_cap)
      : FileStream("mock://short-read"), data_(std::move(data)), cap_(short_read_cap) {}

  size_t Read(void *buf, size_t n) override {
    size_t avail = data_.size() - pos_;
    size_t to_copy = std::min({n, avail, cap_});
    std::memcpy(buf, data_.data() + pos_, to_copy);
    pos_ += to_copy;
    return to_copy;
  }
  void SeekRead(ptrdiff_t, int) override {
    throw std::logic_error("not needed");
  }
  ssize_t TellRead() const override {
    return static_cast<ssize_t>(pos_);
  }
  size_t Size() const override {
    return data_.size();
  }
  void Close() override {}

 private:
  std::string data_;
  size_t pos_ = 0;
  size_t cap_;
};

// Models S3FileStream: Read() delivers up to `n` bytes starting at the current position,
// truncated to whatever remains in the object (a short, non-zero read on the final chunk), and
// throws if asked for a range starting at or past the object's end - it never returns 0.
class RemoteLikeStream : public FileStream {
 public:
  explicit RemoteLikeStream(std::string data)
      : FileStream("mock://remote"), data_(std::move(data)) {}

  size_t Read(void *buf, size_t n) override {
    if (n == 0)
      return 0;
    if (pos_ >= data_.size())
      throw std::runtime_error("mock 416: range starts at or past end of object");
    size_t to_copy = std::min(n, data_.size() - pos_);
    std::memcpy(buf, data_.data() + pos_, to_copy);
    pos_ += to_copy;
    return to_copy;
  }
  void SeekRead(ptrdiff_t, int) override {
    throw std::logic_error("not needed");
  }
  ssize_t TellRead() const override {
    return static_cast<ssize_t>(pos_);
  }
  size_t Size() const override {
    return data_.size();
  }
  void Close() override {}

 private:
  std::string data_;
  size_t pos_ = 0;
};

template <typename Stream>
std::string ReadAllThroughBuf(Stream &stream) {
  FileStreamBuf<16> buf(&stream);  // tiny buffer: forces many underflow() calls
  std::istream in(&buf);
  std::string out((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return out;
}

}  // namespace

TEST(FileStreamBuf, GenuinePartialReadsDoNotTruncate) {
  const std::string payload = MakePayload(200);
  ShortReadStream stream(payload, /*short_read_cap=*/3);  // far smaller than the 16-byte buffer

  std::string result;
  ASSERT_NO_THROW({ result = ReadAllThroughBuf(stream); });
  EXPECT_EQ(result.size(), payload.size());
  EXPECT_EQ(result, payload);
}

TEST(FileStreamBuf, RemoteStreamNeverReadsPastEnd) {
  // Not a multiple of the 16-byte buffer, so the final underflow() lands exactly on the
  // boundary the fix targets.
  const std::string payload = MakePayload(163);
  RemoteLikeStream stream(payload);

  std::string result;
  ASSERT_NO_THROW({ result = ReadAllThroughBuf(stream); });
  EXPECT_EQ(result.size(), payload.size());
  EXPECT_EQ(result, payload);
}

TEST(FileStreamBuf, RemoteStreamExactMultipleOfBuffer) {
  // 32 is an exact multiple of the 16-byte buffer: the boundary case where the very last
  // underflow() call must not fire at all.
  const std::string payload = MakePayload(32);
  RemoteLikeStream stream(payload);

  std::string result;
  ASSERT_NO_THROW({ result = ReadAllThroughBuf(stream); });
  EXPECT_EQ(result.size(), payload.size());
  EXPECT_EQ(result, payload);
}

}  // namespace dali
