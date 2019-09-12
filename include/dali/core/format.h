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

#ifndef DALI_UTIL_MAKE_STRING_H_
#define DALI_UTIL_MAKE_STRING_H_

#include <string>

namespace dali {

template<class Delimiter>
void print_delim(std::ostream&os, const Delimiter &delimiter) {
  // No-op
}


template <class Delimiter, class T>
void print_delim(std::ostream &os, const Delimiter &delimiter, const T& val) {
  os << val;
}

/**
 * Populates stream with given arguments, as long as they have
 * `operator<<` defined for stream operation.
 */
template <class Delimiter, class T, class... Args>
void print_delim(std::ostream &os, const Delimiter &delimiter, const T &val,
                 const Args &... args) {
  os << val << delimiter;
  print_delim(os, delimiter, args...);
}


/**
 * Creates std::string from arguments, as long as every element has `operator<<`
 * defined for stream operation.
 *
 * If there's no `operator<<`, compiler will fire an error, saying:
 * " no match for ‘operator<<’ (operand types are ‘std::stringstream’ and ‘<your-type-here>’) "
 *
 * @param delimiter String, which will separate arguments in the final string
 * @return Concatenated std::string
*/
template <class Delimiter, class... Args>
std::string make_string_delim(const Delimiter &delimiter, const Args &... args) {
  std::stringstream ss;
  print_delim(ss, delimiter, args...);
  return ss.str();
}


/**
 * In case there are no Args to operate on, return empty string.
 */
template <class Delimiter>
std::string make_string_delim(const Delimiter &delimiter) {
  return {};
}


/**
 * Convenient version of @see make_string_delim, which takes a whitespace (' ')
 * as a delimiter.
 */
template <class... Args>
std::string make_string(const Args &... args) {
  return make_string_delim(" ", args...);
}

}  // namespace dali

#endif  // DALI_UTIL_MAKE_STRING_H_
