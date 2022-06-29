// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_STREAM_H_
#define DALI_CORE_STREAM_H_

#include <cstdio>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include "dali/core/api_helper.h"

namespace dali {

using ssize_t = std::make_signed_t<size_t>;

/**
 * An exception thrown when the stream ended before all requested data could be read.
 */
class DLL_PUBLIC EndOfStream : public std::out_of_range {
 public:
  explicit EndOfStream(const char *message = "End of stream") : std::out_of_range(message) {}
};

/**
 * @brief An abstract file-like interface for reading data.
 */
class InputStream {
 public:
  virtual ~InputStream() = default;

  /**
   * @brief Reads all requested data from the stream; if not all of the data can be read,
   *        an exception is thrown.
   *
   * @param buf   the output buffer
   * @param bytes the number of bytes to read
   */
  inline void ReadBytes(void *buf, size_t bytes) {
    char *b = static_cast<char *>(buf);
    while (bytes) {
      ssize_t n = Read(b, bytes);
      if (n == 0)
        throw EndOfStream();
      if (n < 0)
        throw std::runtime_error("An error occurred while reading data.");
      b += n;
      assert(static_cast<size_t>(n) <= bytes);
      bytes -= n;
    }
  }

  /**
   * @brief Reads one object of given type from the stream
   *
   * @tparam T  the type of the object to read; should be trivially copyable or otherwise
   *            safe to be overwritten with memcpy or similar.
   */
  template <typename T>
  inline T ReadOne() {
    T t;
    ReadAll(&t, 1);
    return t;
  }

  /**
   * @brief Reads `count` instances of type T from the stream to the provided buffer
   *
   * If the function cannot read the requested number of objects, an exception is thrown
   *
   * @tparam T    the type of the object to read; should be trivially copyable or otherwise
   *              safe to be overwritten with memcpy or similar.
   * @param buf   the output buffer
   * @param count the number of objects to read
   */
  template <typename T>
  inline void ReadAll(T *buf, size_t count) {
    ReadBytes(buf, sizeof(T) * count);
  }

  /**
   * @brief Skips `count` objects in the stream
   *
   * Skips over the length of `count` objects of given type (by default char,
   * because sizeof(char) == 1).
   *
   * NOTE: Negative skips are allowed.
   *
   * @tparam T type of the object to skip; defaults to `char`
   * @param count the number of objects to skip
   */
  template <typename T = char>
  void Skip(ssize_t count = 1) {
    SeekRead(count * sizeof(T), SEEK_CUR);
  }

  /**
   * @brief Reads data from the stream and advances the read pointer; partial reads are allowed.
   *
   * A valid implementation of this function reads up to `bytes` bytes from the stream and
   * stores them in `buf`. If the function cannot read all of the requested bytes due to
   * end-of-file, it shall read all it can and return the number of bytes actually read.
   * When reading from a regular file and the file pointer is already at the end, the function
   * shall return 0.
   *
   * This function does not throw EndOfStream.
   *
   * @param buf       the output buffer
   * @param bytes     the number of bytes to read
   * @return size _t  the number of bytes actually read or
   *                  0 in case of end-of-stream condition
   */
  virtual size_t Read(void *buf, size_t bytes) = 0;

  /**
   * @brief Moves the read pointer in the stream.
   *
   * If the new pointer would be out of range, the function may either clamp it to a valid range
   * or throw an error.
   *
   * @param pos     the offset to move
   * @param whence  the beginning - SEEK_SET, SEEK_CUR or SEEK_END
   */
  virtual void SeekRead(ptrdiff_t pos, int whence = SEEK_SET) = 0;

  /**
   * @brief Returns the current read position, in bytes from the beginnging, in the stream.
   */
  virtual ssize_t TellRead() const = 0;

  /**
   * @brief Returns the length, in bytes, of the stream.
   */
  virtual size_t Size() const = 0;

  /**
   * @brief Returns the size as a signed integer.
   */
  inline ssize_t SSize() const {
    return Size();
  }
};


class MemInputStream : public InputStream {
 public:
  MemInputStream() = default;
  ~MemInputStream() = default;
  MemInputStream(const void *mem, size_t bytes) {
    start_ = static_cast<const char *>(mem);
    size_ = bytes;
  }

  size_t Read(void *buf, size_t bytes) override {
    ptrdiff_t left = size_ - pos_;
    if (left < static_cast<ptrdiff_t>(bytes))
      bytes = left;
    std::memcpy(buf, start_ + pos_, bytes);
    pos_ += bytes;
    return bytes;
  }

  ssize_t TellRead() const override {
    return pos_;
  }

  void SeekRead(ssize_t offset, int whence = SEEK_SET) override {
    if (whence == SEEK_CUR) {
      offset += pos_;
    } else if (whence == SEEK_END) {
      offset += size_;
    } else {
      assert(whence == SEEK_SET);
    }
    if (offset < 0 || offset > size_)
      throw std::out_of_range("The requested position in the stream is out of range");
    pos_ = offset;
  }

  size_t Size() const override {
    return size_;
  }

 private:
  const char *start_ = nullptr;
  ptrdiff_t size_ = 0;
  ptrdiff_t pos_ = 0;
};

}  // namespace dali

#endif  // DALI_CORE_STREAM_H_
