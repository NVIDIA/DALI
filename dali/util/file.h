// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_UTIL_FILE_H_
#define DALI_UTIL_FILE_H_

#include <cstdio>
#include <string>

#include "dali/common.h"

namespace dali {

class FileStream {
 public:
  static FileStream * Open(const std::string& uri);

  virtual void Close() = 0;
  virtual size_t Read(uint8_t * buffer, size_t n_bytes) = 0;
  virtual void Seek(int64 pos) = 0;
  virtual size_t Size() const = 0;
  virtual ~FileStream() {}

 protected:
  explicit FileStream(const std::string& path) :
    path_(path)
    {}

  std::string path_;
};

}  // namespace dali

#endif  // DALI_UTIL_FILE_H_
