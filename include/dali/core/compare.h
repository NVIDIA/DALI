// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// limitations under the License.#ifndef DALI_MAKE_STRING_H

#ifndef DALI_CORE_COMPARE_H_
#define DALI_CORE_COMPARE_H_

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include "dali/core/float16.h"

namespace dali {

template <typename A, typename B>
constexpr std::enable_if_t<(is_arithmetic_or_half<A>::value || std::is_enum_v<A>) &&
                          (is_arithmetic_or_half<B>::value || std::is_enum_v<B>), int>
compare(const A &a, const B &b) {
  return a < b ? -1 : b < a ? 1 : 0;
}

constexpr int compare(const void *a, const void *b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

inline int compare(const std::string &a, const std::string &b) {
  return a.compare(b);
}

inline int compare(std::string_view a, std::string_view b) {
  return a.compare(b);
}

/** Lexicographical 3-way comparison.
 *
 * Compares tuple elements and returns the sign of the first non-equal comparison.
 * If the tuples have different lengths and the common part compares equal, the shorter tuple
 * is ordered before the longer one.
 */
template <int idx = 0, typename... Ts, typename... Us>
inline int compare(const std::tuple<Ts...> &a, const std::tuple<Us...> &b) {
  if constexpr (idx < sizeof...(Ts) && idx < sizeof...(Us)) {
    if (int cmp = compare(std::get<idx>(a), std::get<idx>(b)))
      return cmp;
    return compare<idx + 1>(a, b);
  } else {
    return compare(sizeof...(Ts), sizeof...(Us));
  }
}

template <typename A, typename B, typename C, typename D>
inline int compare(const std::pair<A, B> &ab, const std::pair<C, D> &cd) {
  if (int cmp = compare(ab.first, cd.first))
    return cmp;
  return compare(ab.second, cd.second);
}

template <typename A, typename B>
int compare_range(A &&a, B &&b) {
  auto i = std::begin(a);
  auto j = std::begin(b);
  auto ae = std::end(a);
  auto be = std::end(b);
  for (; i != ae && j != be; ++i, ++j) {
    if (int cmp = compare(*i, *j))
      return cmp;
  }
  if (i != ae)
    return 1;
  if (j != be)
    return -1;
  return 0;
}

/** Lexicographical 3-way comparison.
 *
 * Compares range elements and returns the sign of the first non-equal comparison.
 * If the ranges have different lengths and the common part compares equal, the shorter range
 * is ordered before the longer one.
 */
template <typename A, typename B,
          typename = decltype(std::begin(std::declval<const A &>())),
          typename = decltype(std::end(std::declval<const A &>())),
          typename = decltype(std::begin(std::declval<const B &>())),
          typename = decltype(std::end(std::declval<const B &>()))>
int compare(const A &a, const B &b) {
  return compare_range(a, b);
}

}  // namespace dali

#endif  // DALI_CORE_COMPARE_H_
