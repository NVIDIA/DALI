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

#ifndef DALI_UTIL_URI_H_
#define DALI_UTIL_URI_H_

#include <cassert>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include "dali/core/api_helper.h"

namespace dali {

class URI {
 private:
  std::string uri_;  // the original URI string
  std::ptrdiff_t scheme_start_ = 0, scheme_end_ = 0;
  std::ptrdiff_t authority_start_ = 0, authority_end_ = 0;
  std::ptrdiff_t path_start_ = 0, path_end_ = 0;
  std::ptrdiff_t query_start_ = 0, query_end_ = 0;
  std::ptrdiff_t fragment_start_ = 0, fragment_end_ = 0;
  bool valid_ = false;
  std::string err_msg_;

  void enforce_valid() const {
    if (!valid_)
      throw std::runtime_error(uri_ + " is not a valid URI: " + err_msg_);
  }

  std::string_view uri_part(ptrdiff_t start, ptrdiff_t end) const {
    enforce_valid();
    assert(end >= start);
    return std::string_view{uri_.c_str() + start, static_cast<size_t>(end - start)};
  }

 public:
  /** Parse options */
  enum ParseOpts : uint32_t {
    Default = 0,
    AllowNonEscaped = 1 << 0,  // 0x0001 - Convenient when we want to check if a URI is valid before
                               // escaping it (e.g. replacing spaces with %20).
    // Add more options as needed
  };

  /**
   * @brief Parses a URI string
   *
   * @param uri URI to parse
   * @param opts option bitmask (see ParseOpts)
   * @return DLL_PUBLIC
   */
  static DLL_PUBLIC URI Parse(std::string uri, ParseOpts opts = ParseOpts::Default);

  bool valid() const {
    return valid_;
  }

  std::string_view scheme() const {
    return uri_part(scheme_start_, scheme_end_);
  }

  std::string_view authority() const {
    return uri_part(authority_start_, authority_end_);
  }

  std::string_view scheme_authority() const {
    return uri_part(scheme_start_, authority_end_);
  }

  std::string_view path() const {
    return uri_part(path_start_, path_end_);
  }

  std::string_view scheme_authority_path() const {
    return uri_part(scheme_start_, path_end_);
  }

  std::string_view query() const {
    return uri_part(query_start_, query_end_);
  }

  std::string_view path_and_query() const {
    return uri_part(path_start_, std::max(path_end_, query_end_));
  }

  std::string_view fragment() const {
    return uri_part(fragment_start_, fragment_end_);
  }
};

}  // namespace dali

#endif  // DALI_UTIL_URI_H_
