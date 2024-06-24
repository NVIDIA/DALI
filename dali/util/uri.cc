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

#include "dali/util/uri.h"
#include <utility>
#include <stdexcept>

namespace dali {

inline bool allowed_scheme_char(char c) {
  return std::isalnum(c) || c == '.' || c == '+' || c == '-';
}

inline bool allowed_char(char c) {
  // See https://en.wikipedia.org/wiki/Uniform_Resource_Identifier
  // gen-delims: :, /, ?, #, [, ], and @
  // sub-delims: !, $, &, ', (, ), *, +, ,, ;, and =
  // unreserved characters (uppercase and lowercase letters, decimal digits, -, ., _, and ~)
  // the character %
  static const std::string gen_delims = ":/?#[]@";
  static const std::string sub_delims = "!$&'()*+,;=";
  static const std::string unreserved = "-._~";
  return (std::isalnum(c)
    || gen_delims.find(c) != std::string::npos
    || sub_delims.find(c) != std::string::npos
    || unreserved.find(c) != std::string::npos);
}

std::string display_char(char c) {
  switch (c) {
    case '\a':
      return "\\a";
    case '\b':
      return "\\b";
    case '\t':
      return "\\t";
    case '\n':
      return "\\n";
    case '\v':
      return "\\v";
    case '\f':
      return "\\f";
    case '\r':
      return "\\r";
    case '\?':
      return "\\\?";
    default:
      return {c};
  }
}

URI URI::Parse(std::string uri, URI::ParseOpts opts) {
  // See https://en.wikipedia.org/wiki/Uniform_Resource_Identifier
  URI parsed;
  parsed.uri_ = std::move(uri);
  parsed.valid_ = true;
  size_t len = parsed.uri_.length();
  const char* p_start = parsed.uri_.data();
  const char* p_end = p_start + len;
  const char* p = p_start;

  // Scheme
  parsed.scheme_start_ = p - p_start;

  if (!std::isalpha(*p)) {
    parsed.valid_ = false;
    parsed.err_msg_ = "First character should be a letter";
    return parsed;
  }
  while (*p != '\0' && *p !=  ':') {
    if (!allowed_scheme_char(*p)) {
      parsed.valid_ = false;
      parsed.err_msg_ = "Invalid character found (" + display_char(*p) + ") in scheme";
      return parsed;
    }
    p++;
  }
  parsed.scheme_end_ = p - p_start;
  if (parsed.scheme_end_ <= parsed.scheme_start_) {
    parsed.valid_ = false;
    parsed.err_msg_ = "Empty scheme";
    return parsed;
  }

  if (*p != ':') {
    parsed.valid_ = false;
    parsed.err_msg_ = "Expected a colon after the URI scheme";
    return parsed;
  }
  p++;

  // Authority
  if (*p == '/' && *(p + 1) == '/') {
    p += 2;
    parsed.authority_start_ = p - p_start;
    while (*p != '\0' && *p !=  '/') {
      if (!allowed_char(*p)) {
        parsed.valid_ = false;
        parsed.err_msg_ = "Invalid character found (" + display_char(*p) + ") in authority";
        return parsed;
      }
      p++;
    }
    parsed.authority_end_ = p - p_start;
  }

  if (*p == '\0')
    return parsed;

  bool allow_non_escaped =
      (opts & URI::ParseOpts::AllowNonEscaped) == URI::ParseOpts::AllowNonEscaped;

  // Path
  parsed.path_start_ = p - p_start;
  while (*p != '\0' && *p !=  '?') {
    if (!allowed_char(*p) && !allow_non_escaped) {
      parsed.valid_ = false;
      parsed.err_msg_ = "Invalid character found (" + display_char(*p) + ") in path";
      return parsed;
    }
    p++;
  }
  parsed.path_end_ = p - p_start;
  if (*p == '\0')
    return parsed;

  // Query
  p++;
  parsed.query_start_ = p - p_start;
  while (*p != '\0' && *p !=  '#') {
    if (!allowed_char(*p) && !allow_non_escaped) {
      parsed.valid_ = false;
      parsed.err_msg_ = "Invalid character found (" + display_char(*p) + ") in query";
      return parsed;
    }
    p++;
  }
  parsed.query_end_ = p - p_start;
  if (*p == '\0')
    return parsed;

  // Fragment
  p++;
  parsed.fragment_start_ = p - p_start;
  while (*p != '\0') {
    if (!allowed_char(*p) && !allow_non_escaped) {
      parsed.valid_ = false;
      parsed.err_msg_ = "Invalid character found (" + display_char(*p) + ") in fragment";
      return parsed;
    }
    p++;
  }
  parsed.fragment_end_ = p - p_start;
  return parsed;
}

}  // namespace dali
