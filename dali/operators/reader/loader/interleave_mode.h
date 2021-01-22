// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_INTERLEAVE_MODE_H_
#define DALI_OPERATORS_READER_LOADER_INTERLEAVE_MODE_H_

#include <string>
#include <cstring>
#include <iostream>
#include "dali/core/format.h"

namespace dali {

enum class InterleaveMode {
  ShortenContinuous = 0,
  RepeatContinuous  = 1,
  ClampContinuous   = 2,
  Shorten           = 3,
  Repeat            = 4,
  Clamp             = 5
};

inline InterleaveMode ParseInterleaveMode(const char *mode) {
  if (!std::strcmp(mode, "shorten_continuous")) {
    return InterleaveMode::ShortenContinuous;
  } else if (!std::strcmp(mode, "repeat_continuous")) {
    return InterleaveMode::RepeatContinuous;
  } else if (!std::strcmp(mode, "clamp_continuous")) {
    return InterleaveMode::ClampContinuous;
  } else if (!std::strcmp(mode, "shorten")) {
    return InterleaveMode::Shorten;
  } else if (!std::strcmp(mode, "repeat")) {
    return InterleaveMode::Repeat;
  } else if (!std::strcmp(mode, "clamp")) {
    return InterleaveMode::Clamp;
  } else {
    DALI_FAIL(make_string("Invalid interleave mode: ", mode));
  }
}

inline InterleaveMode ParseInterleaveMode(const std::string &s) {
  return ParseInterleaveMode(s.c_str());
}

inline std::ostream &operator<<(std::ostream &os, InterleaveMode mode) {
  switch (mode) {
    case InterleaveMode::ShortenContinuous:
      return os << "shorten_continuous";
    case InterleaveMode::RepeatContinuous:
      return os << "repeat_continuous";
    case InterleaveMode::ClampContinuous:
      return os << "clamp_continuous";
    case InterleaveMode::Shorten:
      return os << "shorten";
    case InterleaveMode::Repeat:
      return os << "repeat";
    case InterleaveMode::Clamp:
      return os << "clamp";
    default:
      return os << "(" << static_cast<int>(mode) << ")";
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_INTERLEAVE_MODE_H_
