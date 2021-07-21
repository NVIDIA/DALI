// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_COPY_VECTOR_HELPER_H_
#define DALI_CORE_COPY_VECTOR_HELPER_H_

#include <type_traits>
#include <utility>
#include <vector>

namespace dali {

namespace detail {

template <typename T, typename S>
void copy_vector(std::vector<T> &out, const std::vector<S> &in) {
  out.reserve(in.size());
  out.clear();
  for (decltype(auto) v : in) {
    out.emplace_back(static_cast<T>(v));
  }
}

/** @brief This overload simply forwards the reference */
template <typename T>
std::vector<T> &&convert_vector(std::vector<T> &&v) {
  return std::move(v);
}

/** @brief This overload simply forwards the reference */
template <typename T>
const std::vector<T> &convert_vector(const std::vector<T> &v) {
  return v;
}

/** @brief This overload converts elements from v and returns a vector of converted objects */
template <typename T, typename S>
std::enable_if_t<!std::is_same<T, S>::value, std::vector<T>>
convert_vector(const std::vector<S> &v) {
  std::vector<T> out;
  copy_vector(out, v);
  return out;
}

}  // namespace detail

}  // namespace dali

#endif  // DALI_CORE_COPY_VECTOR_HELPER_H_
