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

#include "ndll/util/local_file.h"
#include "ndll/error_handling.h"

namespace ndll {

LocalFileStream::LocalFileStream(const std::string& path) :
  FileStream(path) {
  fp_ = std::fopen(path.c_str(), "rb");
  NDLL_ENFORCE(fp_ != nullptr, "Could not open file " + path + ": " + std::strerror(errno));
}

void LocalFileStream::Close() {
  if (fp_ != nullptr) {
    std::fclose(fp_);
  }
  fp_ = nullptr;
}

void LocalFileStream::Seek(int64 pos) {
#ifndef _MSC_VER
    NDLL_ENFORCE(!std::fseek(fp_, pos, SEEK_SET),
      "Seek operation did not succeed: " + std::string(std::strerror(errno)) );
#else
    NDLL_ENFORCE(!_fseeki64(fp_, pos, SEEK_SET),
      "Seek operation did not succeed: " + std::string(std::strerror(errno)) );
#endif
}

void LocalFileStream::Read(uint8_t * buffer, size_t n_bytes) {
  size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
  NDLL_ENFORCE(n_read == n_bytes, "Error reading from a file");
}

size_t LocalFileStream::Size() const {
  struct stat sb;
  if (stat(path_.c_str(), &sb) == -1) {
    NDLL_FAIL("Unable to stat file " + path_ + ": " + std::strerror(errno));
  }
  return sb.st_size;
}

}  // namespace ndll;
