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
// limitations under the License.#ifndef DALI_MAKE_STRING_H

#ifndef DALI_CORE_FORMAT_H_
#define DALI_CORE_FORMAT_H_

#include <string>
#include <sstream>

namespace dali {

struct no_delimiter {};

inline std::ostream &operator<<(std::ostream &os, no_delimiter) { return os; }

template <typename Delimiter>
void print_delim(std::ostream &os, const Delimiter &delimiter) {
  // No-op
}


template <typename Delimiter, typename T>
void print_delim(std::ostream &os, const Delimiter &delimiter, const T &val) {
  os << val;
}

/**
 * @brief Populates stream with given arguments, as long as they have
 * `operator<<` defined for stream operation
 */
template <typename Delimiter, typename T, typename... Args>
void print_delim(std::ostream &os, const Delimiter &delimiter, const T &val,
                 const Args &... args) {
  os << val << delimiter;
  print_delim(os, delimiter, args...);
}

/**
 * @brief Populates stream with given arguments, as long as they have
 * `operator<<` defined for stream operation
 */
template <typename... Args>
void print(std::ostream &os, const Args &... args) {
  print_delim(os, no_delimiter(), args...);
}

/**
 * Creates std::string from arguments, as long as every element has `operator<<`
 * defined for stream operation.
 *
 * If there's no `operator<<`, compiler will fire an error
 *
 * @param delimiter String, which will separate arguments in the final string
 * @return Concatenated std::string
*/
template <typename Delimiter, typename... Args>
std::string make_string_delim(const Delimiter &delimiter, const Args &... args) {
  std::stringstream ss;
  print_delim(ss, delimiter, args...);
  return ss.str();
}


/**
 * This overload handles the edge case when no arguments are given and returns an empty string
 */
template <typename Delimiter>
std::string make_string_delim(const Delimiter &) {
  return {};
}

/**
 * @brief Prints args to a string, without any delimiter
 */
template <typename... Args>
std::string make_string(const Args &... args) {
  std::stringstream ss;
  print(ss, args...);
  return ss.str();
}

}  // namespace dali

#endif  // DALI_CORE_FORMAT_H_
