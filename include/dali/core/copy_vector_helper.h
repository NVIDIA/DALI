// Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/small_vector.h"

namespace dali {

namespace detail {

template <typename From, typename To>
struct is_explicitly_convertible {
 private:
  static void sink_cast_target(To);

  template <typename F, typename T>
  constexpr static auto test(int)  // NOLINT
      -> decltype(sink_cast_target(static_cast<T>(std::declval<F>())), true) {
    return true;
  }

  template <typename F, typename T>
  constexpr static bool test(...) {
    return false;
  }

 public:
  constexpr static bool value = test<From, To>(0);  // = requires(From t) { static_cast<To>(t); };
};

template <typename C1, typename C2>
void copy_vector(C1 &out, const C2 &in) {
  using To = typename C1::value_type;
  using From = typename C2::value_type;
  static_assert(is_explicitly_convertible<From, To>::value, "Element types not convertible");
  out.reserve(in.size());
  out.clear();
  for (decltype(auto) v : in) {
    out.emplace_back(static_cast<To>(v));
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
