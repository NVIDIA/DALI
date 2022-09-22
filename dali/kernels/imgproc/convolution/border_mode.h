// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_BORDER_MODE_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_BORDER_MODE_H_

#include <exception>
#include <string>

#include "dali/core/error_handling.h"
#include "dali/core/format.h"

namespace dali {
/**
 * @brief Supported border modes
 */
enum FilterBorderMode {
  DALI_BORDER_REFLECT_101 = 0,
  DALI_BORDER_REFLECT_1001 = 1,
  DALI_BORDER_WRAP = 2,
  DALI_BORDER_REPLICATE = 3,
  DALI_BORDER_FILL = 4,
  DALI_BORDER_VALID = 5
};

inline std::string to_string(const FilterBorderMode& border_mode) {
  switch (border_mode) {
    case DALI_BORDER_REFLECT_101:
      return "reflect_101";
    case DALI_BORDER_REFLECT_1001:
      return "reflect_1001";
    case DALI_BORDER_WRAP:
      return "wrap";
    case DALI_BORDER_REPLICATE:
      return "replicate";
    case DALI_BORDER_FILL:
      return "fill";
    case DALI_BORDER_VALID:
      return "valid";
    default:
      return "<unknown>";
  }
}

inline FilterBorderMode parse(const std::string& border_mode) {
  if (border_mode == "reflect_101")
    return DALI_BORDER_REFLECT_101;
  if (border_mode == "reflect_1001")
    return DALI_BORDER_REFLECT_1001;
  if (border_mode == "wrap")
    return DALI_BORDER_WRAP;
  if (border_mode == "replicate")
    return DALI_BORDER_REPLICATE;
  if (border_mode == "fill")
    return DALI_BORDER_FILL;
  if (border_mode == "valid")
    return DALI_BORDER_VALID;
  throw std::runtime_error(
      make_string("Unknown ``border_mode`` was specified: ``", border_mode, "``."));
}

}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_BORDER_MODE_H_
