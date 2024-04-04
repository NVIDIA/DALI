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
// limitations under the License.

#ifndef DALI_CORE_SOURCE_LOCATION_H_
#define DALI_CORE_SOURCE_LOCATION_H_

#include <array>
#include <cstddef>
#include <type_traits>
#include "dali/core/host_dev.h"

namespace dali {

/**
 * @brief Backport of std::source_location from C++20, using compiler __builtins present in earlier
 * versions. Allows to replace __FILE__ and __LINE__ macros with function calls.
 *
 * Call `source_location::current()` to get the current source location.
 * In most cases it should be used as a default argument to a function, to record the source
 * location of a function call, like so:
 *    void foo(source_location loc = source_location::current()) { ... }
 *
 */
class source_location {
 public:
  DALI_HOST_DEV constexpr source_location() = default;
  DALI_HOST_DEV constexpr source_location(const source_location &) = default;
  DALI_HOST_DEV constexpr source_location &operator=(const source_location &) = default;
  DALI_HOST_DEV constexpr source_location(source_location &&) = default;
  DALI_HOST_DEV constexpr source_location &operator=(source_location &&) = default;


  DALI_HOST_DEV constexpr const char *source_file() const {
    return source_file_;
  }

  DALI_HOST_DEV constexpr const char *function_name() const {
    return function_name_;
  }

  DALI_HOST_DEV constexpr int line() const {
    return line_;
  }

  /**
   * @brief Get the current source location.
   * The caller of this function should not override the default arguments.
   */
  DALI_HOST_DEV constexpr static source_location current(
      const char *source_file = __builtin_FILE(), const char *function_name = __builtin_FUNCTION(),
      int line_ = __builtin_LINE()) {
    return {source_file, function_name, line_};
  }

 private:
  DALI_HOST_DEV constexpr source_location(const char *source_file, const char *function_name,
                                          int line)
      : source_file_(source_file), function_name_(function_name), line_(line) {}
  const char *source_file_ = "";
  const char *function_name_ = "";
  int line_ = 0;
};


}  // namespace dali

#endif  // DALI_CORE_SOURCE_LOCATION_H_
