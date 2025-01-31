// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_ERROR_HANDLING_H_
#define DALI_C_API_2_ERROR_HANDLING_H_

#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/core/error_handling.h"

inline std::ostream &operator<<(std::ostream &os, daliResult_t result) {
  const char *e = daliGetErrorName(result);
  if (e[0] == '<')
    os << "<unknown: " << static_cast<int>(result) << ">";
  else
    os << e;
  return os;
}

inline std::string to_string(daliResult_t result) {
  std::stringstream ss;
  ss << result;
  return ss.str();
}

namespace dali {
namespace c_api {

DLL_PUBLIC daliResult_t HandleError(std::exception_ptr ex);
DLL_PUBLIC daliResult_t CheckInit();

class InvalidHandle : public std::invalid_argument {
 public:
  InvalidHandle() : std::invalid_argument("The handle is invalid") {}
  explicit InvalidHandle(const std::string &what) : std::invalid_argument(what) {}
  explicit InvalidHandle(const char *what) : std::invalid_argument(what) {}
};

inline InvalidHandle NullHandle() { return InvalidHandle("The handle must not be NULL."); }

inline InvalidHandle NullHandle(const char *what_handle) {
  return InvalidHandle(make_string("The ", what_handle, " handle must not be NULL."));
}

}  // namespace c_api
}  // namespace dali

#define DALI_PROLOG() try { if (auto err = dali::c_api::CheckInit()) return err; else;
#define DALI_EPILOG() return DALI_SUCCESS; } catch (...) {     \
  return ::dali::c_api::HandleError(std::current_exception()); \
}

#endif  // DALI_C_API_2_ERROR_HANDLING_H_
