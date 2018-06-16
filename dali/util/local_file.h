// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_UTIL_LOCAL_FILE_H_
#define DALI_UTIL_LOCAL_FILE_H_

#include <cstdio>
#include <string>

#include "dali/common.h"
#include "dali/util/file.h"

namespace dali {

class LocalFileStream : public FileStream {
 public:
  explicit LocalFileStream(const std::string& path);
  void Close() override;
  size_t Read(uint8_t * buffer, size_t n_bytes) override;
  void Seek(int64 pos) override;
  size_t Size() const override;

  ~LocalFileStream() override {
    Close();
  }

 private:
  FILE * fp_;
};

}  // namespace dali

#endif  // DALI_UTIL_LOCAL_FILE_H_
