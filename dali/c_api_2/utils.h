// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_UTILS_H_
#define DALI_C_API_2_UTILS_H_

#include <optional>
#include <stdexcept>
#include <string_view>

namespace dali::c_api {

template <typename T>
std::optional<T> ToOptional(const T *nullable) {
  if (nullable == nullptr)
    return std::nullopt;
  else
    return *nullable;
}

}  // namespace dali::c_api

#endif  // DALI_C_API_2_UTILS_H_
