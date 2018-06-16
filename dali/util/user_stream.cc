// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <mutex>

#include "dali/util/user_stream.h"

namespace dali {

  UserStream * UserStream::us_ = nullptr;
  std::mutex UserStream::m_;

}  // namespace dali
