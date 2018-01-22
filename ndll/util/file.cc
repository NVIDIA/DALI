// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <string>

#include "ndll/util/file.h"
#include "ndll/util/local_file.h"

namespace ndll {

FileStream * FileStream::Open(const std::string& uri) {
  if (uri.find("file://") == 0) {
    return new LocalFileStream(uri.substr(std::string("file://").size()));
  } else {
    return new LocalFileStream(uri);
  }
}
}  // namespace ndll
