// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <mutex>

#include "ndll/util/user_stream.h"

namespace ndll {

  UserStream * UserStream::us_ = nullptr;
  std::mutex UserStream::m_;

}  // namespace ndll
