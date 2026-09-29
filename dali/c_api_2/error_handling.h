// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/dali.h"
#include "dali/c_api_2/pipeline_registry.h"
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

/** Thrown when an operation is attempted while DALI is shutting down or has been shut down. */
class Unloading : public std::runtime_error {
 public:
  Unloading() : std::runtime_error("DALI is unloading") {}
  explicit Unloading(const std::string &what) : std::runtime_error(what) {}
  explicit Unloading(const char *what) : std::runtime_error(what) {}
};

inline InvalidHandle NullHandle() { return InvalidHandle("The handle must not be NULL."); }

inline InvalidHandle NullHandle(const char *what_handle) {
  return InvalidHandle(make_string("The ", what_handle, " handle must not be NULL."));
}

}  // namespace c_api
}  // namespace dali

#define DALI_PROLOG() try { ::dali::c_api::ActiveCallGuard dali_active_call_guard_; \
  if (auto err = dali::c_api::CheckInit()) return err; else;  // NOLINT(readability/braces)
#define DALI_EPILOG() return DALI_SUCCESS; } catch (...) {     \
  return ::dali::c_api::HandleError(std::current_exception()); \
}

#endif  // DALI_C_API_2_ERROR_HANDLING_H_
