// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_COLOR_SPACE_CONVERSION_UTILS_H_
#define DALI_UTIL_COLOR_SPACE_CONVERSION_UTILS_H_

#include "dali/core/convert.h"

namespace dali {

template <typename OutputType, typename InputType>
inline static OutputType Y(InputType r, InputType g, InputType b) {
  return ConvertSatNorm<OutputType>(
    static_cast<InputType>(0.257f * r + 0.504f * g + 0.098f * b + 16.0f));
}

template <typename OutputType, typename InputType>
inline static OutputType Cb(InputType r, InputType g, InputType b) {
  return (r == g && g == b) ? ConvertNorm<OutputType>(0.5f) :
    ConvertSatNorm<OutputType>(
      static_cast<InputType>(-0.148f * r - 0.291f * g + 0.439f * b + 128.0f));
}

template <typename OutputType, typename InputType>
inline static OutputType Cr(InputType r, InputType g, InputType b) {
  return (r == g && g == b) ? ConvertNorm<OutputType>(0.5f) :
    ConvertSatNorm<OutputType>(
      static_cast<InputType>(0.439f * r - 0.368f * g - 0.071f * b + 128.0f));
}

template <typename OutputType, typename InputType>
inline static OutputType GrayScale(InputType r, InputType g, InputType b) {
  return (r == g && g == b) ? ConvertNorm<OutputType>(r) :
    ConvertSatNorm<OutputType>(
      static_cast<InputType>(0.299f * r + 0.587f * g + 0.114f * b));
}

}  // namespace dali

#endif  // DALI_UTIL_COLOR_SPACE_CONVERSION_UTILS_H_
