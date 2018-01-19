// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_FILE_H_
#define NDLL_UTIL_FILE_H_

#include <cstdio>
#include <string>

#include "ndll/common.h"

namespace ndll {

class FileStream {
 public:
  static FileStream * Open(const std::string& uri);

  virtual void Close() = 0;
  virtual void Read(uint8_t * buffer, size_t n_bytes) = 0;
  virtual void Seek(int64 pos) = 0;
  virtual size_t Size() const = 0;
 protected:
  explicit FileStream(const std::string& path) :
    path_(path)
    {}
  virtual ~FileStream() {}

  std::string path_;
};

}  // namespace ndll

#endif  // NDLL_UTIL_FILE_H_
