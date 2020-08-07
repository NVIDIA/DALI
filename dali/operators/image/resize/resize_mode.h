// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_MODE_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_MODE_H_

#include <string>
#include <cstring>
#include <iostream>
#include "dali/core/format.h"

namespace dali {

enum class ResizeMode {
    /// Resize exactly to this size; missing extents are calculated to maintain aspect ratio
    Default = 0,
    /// Resize exactly to this size; missing extents keep input size
    Stretch = 1,
    /// Maintain aspect ratio; resized image is not larger than requested in any dimension
    NotLarger = 2,
    /// Maintain aspect ratio; resized image is not smaller than requested in any dimension
    NotSmaller = 3,
};

inline ResizeMode ParseResizeMode(const char *mode) {
  if (!std::strcmp(mode, "default")) {
    return ResizeMode::Default;
  } else if (!std::strcmp(mode, "stretch")) {
    return ResizeMode::Stretch;
  } else if (!std::strcmp(mode, "not_larger")) {
    return ResizeMode::NotLarger;
  } else if (!std::strcmp(mode, "not_smaller")) {
    return ResizeMode::NotSmaller;
  } else {
    DALI_FAIL(make_string("Invalid resize mode: ", mode));
  }
}

inline ResizeMode ParseResizeMode(const std::string &s) {
  return ParseResizeMode(s.c_str());
}

inline std::ostream &operator<<(std::ostream &os, ResizeMode mode) {
  switch (mode) {
    case ResizeMode::Default:
      return os << "default";
    case ResizeMode::Stretch:
      return os << "stretch";
    case ResizeMode::NotLarger:
      return os << "not_larger";
    case ResizeMode::NotSmaller:
      return os << "not smaller";
    default:
      return os << "(" << static_cast<int>(mode) << ")";
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_MODE_H_
