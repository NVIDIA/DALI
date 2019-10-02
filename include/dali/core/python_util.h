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
// limitations under the License.

#ifndef DALI_CORE_PYTHON_UTIL_H_
#define DALI_CORE_PYTHON_UTIL_H_

#include <string>
#include <sstream>
#include <map>
#include "dali/core/util.h"

namespace dali {

inline void escape_string(std::ostream &os, const char *str) {
  const char hex[] = "0123456789abcdef";
  while (unsigned char c = *str++) {
    switch (c) {
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      case '\\':
        os << "\\\\";
        break;
      case '\'':
        os << "\\\'";
        break;
      case '\"':
        os << "\\\"";
        break;
      default:
        if (c >= 32 && c < 128)
          os << c;
        else
          os << "\\x" << hex[c >> 4] << hex[c&15];
        break;
    }
  }
}

inline void escape_string(std::ostream &os, const std::string &s) {
  escape_string(os, s.c_str());
}

template <typename T>
inline void python_repr(std::ostream &os, const T &obj) {
  os << obj;
}

inline void python_repr(std::ostream &os, const char *s) {
  os.put('\'');
  escape_string(os, s);
  os.put('\'');
}

inline void python_repr(std::ostream &os, const std::string &s) {
  python_repr(os, s.c_str());
}

template <typename T>
if_iterable<T> python_repr(std::ostream &os, const T &collection) {
  os << "[";
  bool first = true;
  for (const auto &v : collection) {
    if (!first)
      os << ", ";
    first = false;

    python_repr(os, v);
  }
  os << "]";
}

template <typename T>
inline std::string python_repr(const T &obj) {
  std::stringstream ss;
  python_repr(ss, obj);
  return ss.str();
}

}  // namespace dali

#endif  // DALI_CORE_PYTHON_UTIL_H_
