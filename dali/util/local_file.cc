// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <errno.h>
#include <sys/stat.h>
#include <string>
#include <cstdio>
#include <cstring>

#ifdef _MSC_VER
#include <Windows.h>
#define stat _stat64
#endif

#include "dali/util/local_file.h"
#include "dali/error_handling.h"

namespace dali {

LocalFileStream::LocalFileStream(const std::string& path) :
  FileStream(path) {
  fp_ = std::fopen(path.c_str(), "rb");
  DALI_ENFORCE(fp_ != nullptr, "Could not open file " + path + ": " + std::strerror(errno));
}

void LocalFileStream::Close() {
  if (fp_ != nullptr) {
    std::fclose(fp_);
  }
  fp_ = nullptr;
}

void LocalFileStream::Seek(int64 pos) {
#ifndef _MSC_VER
    DALI_ENFORCE(!std::fseek(fp_, pos, SEEK_SET),
      "Seek operation did not succeed: " + std::string(std::strerror(errno)) );
#else
    DALI_ENFORCE(!_fseeki64(fp_, pos, SEEK_SET),
      "Seek operation did not succeed: " + std::string(std::strerror(errno)) );
#endif
}

size_t LocalFileStream::Read(uint8_t * buffer, size_t n_bytes) {
  size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
  return n_read;
}

size_t LocalFileStream::Size() const {
  struct stat sb;
  if (stat(path_.c_str(), &sb) == -1) {
    DALI_FAIL("Unable to stat file " + path_ + ": " + std::strerror(errno));
  }
  return sb.st_size;
}

}  // namespace dali
