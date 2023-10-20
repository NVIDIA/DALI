// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_VERSION_UTIL_H_
#define DALI_CORE_VERSION_UTIL_H_

namespace dali {

/// @brief Returns the product of all elements in shape
/// @param getter - function obtaining property
/// @param value - value where to put the property, in case of failure it gets `invalid_value`
/// @param property - enum for the select right property
/// @param success_status - enum value for success status of getter
/// @param invalid_value - the value returned when the getter fails
template <typename F, typename V, typename E, typename S>
void GetVersionProperty(F getter, V *value, E property, S success_status, V invalid_value = -1) {
  if (getter(property, value) != success_status) {
    *value = invalid_value;
  }
}

// gets single int that can be represented as int value
constexpr int MakeVersionNumber(int major, int minor, int patch = 0) {
  if (major < 0 || minor < 0 || patch < 0) {
    return -1;
  }
  return major*1000 + minor*10 + patch;
}

}  // namespace dali

#endif  // DALI_CORE_VERSION_UTIL_H_
