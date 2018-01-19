// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_LOCAL_FILE_H_
#define NDLL_UTIL_LOCAL_FILE_H_

#include <cstdio>
#include <string>

#include "ndll/common.h"
#include "ndll/util/file.h"

namespace ndll {

class LocalFileStream : public FileStream {
 public:
  LocalFileStream(const std::string& path);
  void Close() override;
  void Read(uint8_t * buffer, size_t n_bytes) override;
  void Seek(int64 pos) override;
  size_t Size() const override;

 private:
  FILE * fp_;
};

}  // namespace ndll

#endif  // NDLL_UTIL_LOCAL_FILE_H_
