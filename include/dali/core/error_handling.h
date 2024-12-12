// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_ERROR_HANDLING_H_
#define DALI_CORE_ERROR_HANDLING_H_

#include "dali/core/format.h"

#ifndef _MSC_VER
  #if defined(__AARCH64_QNX__) || defined(__AARCH64_GNU__) || defined(__aarch64__)
     #define DALI_USE_STACKTRACE 0
  #else
     #define DALI_USE_STACKTRACE 1
  #endif
#endif  // _MSC_VER

#if DALI_USE_STACKTRACE
#include <cxxabi.h>
#include <execinfo.h>
#endif  // DALI_USE_STACKTRACE

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "dali/core/common.h"

namespace dali {

/**
 * @brief Error object returned by dali functions. If an error is returned ('DALIError'),
 * a string explaining the error can be found by calling 'DALIGetLastError'
 */
enum DALIError_t {
  DALISuccess,
  DALIError,
  DALIErrorCUDA
};

/**
 * @brief Returns a string explaining the last error that occured. Calling this function
 * clears the error. If no error has occured (or it has been wiped out by a previous call
 * to this function), this function returns an empty string
 */
DLL_PUBLIC string DALIGetLastError();

// Sets the error string. Used internally by DALI to pass error strings out to the user
DLL_PUBLIC void DALISetLastError(const string &error_str);

// Appends additional info to last error. Used internally by DALI to pass error
// strings out to the user
DLL_PUBLIC void DALIAppendToLastError(const string &error_str);

class DALIException : public std::runtime_error {
 public:
  explicit DALIException(const std::string &message, const std::string file_and_line = "",
                         const std::string &cpp_backtrace = "")
      : std::runtime_error(message), file_and_line_(file_and_line), cpp_backtrace_(cpp_backtrace) {}

  const std::string &GetFileAndLine() const {
    return file_and_line_;
  }

  const std::string &GetCppBacktrace() const {
    return cpp_backtrace_;
  }

 private:
  std::string file_and_line_;
  std::string cpp_backtrace_;
};

struct unsupported_exception : std::runtime_error {
  explicit unsupported_exception(const std::string &str) : runtime_error(str) {}
};

/** An exception thrown when an invalid dictionary key is provided
 *
 * The exception denotes an invalid key. It can be thrown when:
 * - the key is not found and the function returns a non-nullable type
 * - the key doesn't meet some constraints (e.g. a dictionary doesn't accept an empty
 *   string as a key).
 *
 * This exception is used at the Python boundary to raise KeyError rather than IndexError.
 */
struct invalid_key : std::out_of_range {
  explicit invalid_key(const std::string &message) : std::out_of_range(message) {}
  explicit invalid_key(const char *message) : std::out_of_range(message) {}
};

inline string BuildErrorString(string statement, string file, int line) {
  string line_str = std::to_string(line);
  string error = "[" + std::move(file) + ":" + std::move(line_str) +
    "]: Assert on \"" + std::move(statement) +
    "\" failed";
  return error;
}

#if DALI_USE_STACKTRACE
inline void ltrim(std::string *s) {
    s->erase(s->begin(), std::find_if(s->begin(), s->end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

inline void rtrim(std::string *s) {
    s->erase(std::find_if(s->rbegin(), s->rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s->end());
}

inline void trim(std::string *s) {
    ltrim(s);
    rtrim(s);
}

inline dali::string GetStacktrace() {
  const int MAX_STACK_SIZE = 100;
  void * stack[MAX_STACK_SIZE];
  int nframes = backtrace(stack, MAX_STACK_SIZE);
  dali::string ret = "\nStacktrace (" + std::to_string(nframes) + " entries):\n";
  char **msgs = backtrace_symbols(stack, nframes);
  if (msgs != nullptr) {
    for (int frame = 0; frame < nframes; ++frame) {
      dali::string msg(msgs[frame]);
      size_t symbol_start = string::npos;
      size_t symbol_end = string::npos;
      dali::string s = msgs[frame];
      if ( ((symbol_start = msg.find("_Z")) != string::npos)
          && (symbol_end = msg.find("+0x", symbol_start)) != string::npos ) {
        string left(msg, 0, symbol_start);
        string symbol(msg, symbol_start, symbol_end - symbol_start);
        trim(&symbol);
        string right(msg, symbol_end);
        int status = 0;
        char * demangled_symbol =
          abi::__cxa_demangle(symbol.c_str(), 0, 0, &status);
        if (demangled_symbol != nullptr) {
          s = left + demangled_symbol + right;
          std::free(demangled_symbol);
        }
      }
      ret += "[frame " + std::to_string(frame) + "]: " + s + "\n";
    }
  }
  free(msgs);
  return ret;
}
#else
inline dali::string GetStacktrace() {
  return "";
}
#endif  // DALI_USE_STACKTRACE && DALI_DEBUG

#define ASRT_1(code)                                                          \
  do {                                                                        \
    if (!(code)) {                                                            \
      dali::string error = dali::BuildErrorString(#code, __FILE__, __LINE__); \
      DALISetLastError(error);                                                \
      return DALIError;                                                       \
    }                                                                         \
  } while (0)

#define ASRT_2(code, str)                                                     \
  do {                                                                        \
    if (!(code)) {                                                            \
      dali::string error = dali::BuildErrorString(#code, __FILE__, __LINE__); \
      dali::string usr_str = str;                                             \
      error += ": " + usr_str;                                                \
      DALISetLastError(error);                                                \
      return DALIError;                                                       \
    }                                                                         \
  } while (0)

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define DALI_ASSERT(...) GET_MACRO(__VA_ARGS__, ASRT_2, ASRT_1)(__VA_ARGS__)

#define DALI_FORWARD_ERROR(code) \
  if ((code) == DALIError) {     \
    return DALIError;            \
  }

//////////////////////////////////////////////////////
/// Error checking utilities for the DALI pipeline ///
//////////////////////////////////////////////////////

// For calling DALI library functions
#define DALI_CALL(code)                                         \
  do {                                                          \
    DALIError_t status = code;                                  \
    if (status != DALISuccess) {                                \
      dali::string error = DALIGetLastError();                  \
      DALI_FAIL(error);                                         \
    }                                                           \
  } while (0)

// For calling DALI library functions with extra debug log
#define DALI_CALL_EX(code, extra_info)                          \
  do {                                                          \
    DALIError_t status = code;                                  \
    if (status != DALISuccess) {                                \
      DALIAppendToLastError(extra_info);                        \
      dali::string error = DALIGetLastError();                  \
      DALI_FAIL(error);                                         \
    }                                                           \
  } while (0)

// Excpetion throwing checks for pipeline code
#define ENFRC_1(code)                                                         \
  do {                                                                        \
    if (!(code)) {                                                            \
      dali::string error = dali::string("Assert on \"") + #code +"\" failed"; \
      DALI_FAIL(error);                                                       \
    }                                                                         \
  } while (0)

#define ENFRC_2(code, str)                                                    \
  do {                                                                        \
    if (!(code)) {                                                            \
      dali::string error = dali::string("Assert on \"") + #code +"\" failed"; \
      dali::string usr_str = str;                                             \
      error += ": " + usr_str;                                                \
      DALI_FAIL(error);                                                       \
    }                                                                         \
  } while (0)

#define DALI_ENFORCE(...) GET_MACRO(__VA_ARGS__, ENFRC_2, ENFRC_1)(__VA_ARGS__)

// Enforces that the value of 'var' is in the range [lower, upper)
#define DALI_ENFORCE_IN_RANGE(var, lower, upper)                                         \
  do {                                                                                   \
    if (((var) < (lower)) || (static_cast<size_t>(var) >= static_cast<size_t>(upper))) { \
      dali::string error = "Index " + std::to_string(var) + " out of range [" +          \
        std::to_string(lower) + ", " + std::to_string(upper) + ").";                     \
      DALI_FAIL(error);                                                                  \
    }                                                                                    \
  } while (0)

// Enforces that the input var is in the range [0, upper)
#define DALI_ENFORCE_VALID_INDEX(var, upper) \
  DALI_ENFORCE_IN_RANGE(var, 0, upper)

#define DALI_STR2(x) #x
#define DALI_STR(x) DALI_STR2(x)
#define FILE_AND_LINE __FILE__ ":" DALI_STR(__LINE__)

#define DALI_MESSAGE_AND_STACKTRACE(...)\
  (std::string("[" FILE_AND_LINE "] ")), ##__VA_ARGS__, dali::GetStacktrace()

#define DALI_MESSAGE(...)\
  (std::string("[" FILE_AND_LINE "] ")), ##__VA_ARGS__

#define DALI_FAIL(...)                     \
    throw dali::DALIException(             \
      dali::make_string(__VA_ARGS__),      \
      std::string("[" FILE_AND_LINE "] "), \
      dali::GetStacktrace());

#define DALI_ERROR(...)                                                                    \
  do {                                                                                     \
    std::cerr << dali::make_string(DALI_MESSAGE_AND_STACKTRACE(__VA_ARGS__)) << std::endl; \
  } while (0)

#define DALI_WARN(...)                                                      \
  do {                                                                      \
    std::cerr << dali::make_string(DALI_MESSAGE(__VA_ARGS__)) << std::endl; \
  } while (0)

#define DALI_WARN_ONCE(str)                                                  \
  do {                                                                       \
    static int dummy =                                                       \
        (std::cerr << dali::make_string(DALI_MESSAGE(str)) << std::endl, 0); \
  } while (0)

DLL_PUBLIC void DALIReportFatalProblem(const char *file, int line, const char *pComment);
#define REPORT_FATAL_PROBLEM(comment) DALIReportFatalProblem(__FILE__, __LINE__, comment)

}  // namespace dali

#endif  // DALI_CORE_ERROR_HANDLING_H_
