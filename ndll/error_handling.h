// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_ERROR_HANDLING_H_
#define NDLL_ERROR_HANDLING_H_

#ifndef _MSC_VER
#define NDLL_USE_STACKTRACE 1
#endif  // _MSC_VER

#if NDLL_USE_STACKTRACE
#include <cxxabi.h>
#include <execinfo.h>
#endif  // NDLL_USE_STACKTRACE

#include <cuda_runtime_api.h>
#include <nvml.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "ndll/common.h"
#include "ndll/util/npp.h"

namespace ndll {

/**
 * @brief Error object returned by ndll functions. If an error is returned ('NDLLError'),
 * a string explaining the error can be found by calling 'NDLLGetLastError'
 */
enum NDLLError_t {
  NDLLSuccess = 0,
  NDLLError = 1
};

/**
 * @brief Returns a string explaining the last error that occured. Calling this function
 * clears the error. If no error has occured (or it has been wiped out by a previous call
 * to this function), this function returns an empty string
 */
string NDLLGetLastError();

// Sets the error string. Used internally by NDLL to pass error strings out to the user
void NDLLSetLastError(string error_str);

inline string BuildErrorString(string statement, string file, int line) {
  string line_str = std::to_string(line);
  string error = "[" + file + ":" + line_str +
    "]: Assert on \"" + statement +
    "\" failed";
  return error;
}

#define ASRT_1(code)                                                          \
  do {                                                                        \
    if (!(code)) {                                                            \
      ndll::string error = ndll::BuildErrorString(#code, __FILE__, __LINE__); \
      NDLLSetLastError(error);                                                \
      return NDLLError;                                                       \
    }                                                                         \
  } while (0)

#define ASRT_2(code, str)                                                     \
  do {                                                                        \
    if (!(code)) {                                                            \
      ndll::string error = ndll::BuildErrorString(#code, __FILE__, __LINE__); \
      ndll::string usr_str = str;                                             \
      error += ": " + usr_str;                                                \
      NDLLSetLastError(error);                                                \
      return NDLLError;                                                       \
    }                                                                         \
  } while (0)

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define NDLL_ASSERT(...) GET_MACRO(__VA_ARGS__, ASRT_2, ASRT_1)(__VA_ARGS__)

#define NDLL_FORWARD_ERROR(code) \
  if ((code) == NDLLError) {     \
    return NDLLError;            \
  }

#define NDLL_RETURN_ERROR(str)                                            \
  do {                                                                    \
    ndll::string file = __FILE__;                                         \
    ndll::string line = std::to_string(__LINE__);                         \
    ndll::string error =  "[" + file + ":" + line + "]: Error in NDLL: "; \
    error += str;                                                         \
    NDLLSetLastError(error);                                              \
    return NDLLError;                                                     \
  } while (0)

// For checking npp return errors in ndll library functions
#define NDLL_CHECK_NPP(code)                              \
  do {                                                    \
    NppStatus status = code;                              \
    if (status != NPP_SUCCESS) {                          \
    ndll::string file = __FILE__;                         \
      ndll::string line = std::to_string(__LINE__);       \
      ndll::string error = "[" + file + ":" + line +      \
        "]: NPP error \"" +                               \
        nppErrorString(status) + "\"";                    \
      NDLLSetLastError(error);                            \
      return NDLLError;                                   \
    }                                                     \
  } while (0)

//////////////////////////////////////////////////////
/// Error checking utilities for the NDLL pipeline ///
//////////////////////////////////////////////////////

// For calling CUDA library functions
#define CUDA_CALL(code)                                   \
  do {                                                    \
    cudaError_t status = code;                            \
    if (status != cudaSuccess) {                          \
      ndll::string file = __FILE__;                       \
      ndll::string line = std::to_string(__LINE__);       \
      ndll::string error = "[" + file + ":" + line +      \
        "]: CUDA error \"" +                              \
        cudaGetErrorString(status) + "\"";                \
      throw std::runtime_error(error);                    \
    }                                                     \
  } while (0)

// For calling NVML library functions
#define NVML_CALL(code)                                   \
  do {                                                    \
    nvmlReturn_t status = code;                           \
    if (status != NVML_SUCCESS) {                         \
      ndll::string file = __FILE__;                       \
      ndll::string line = std::to_string(__LINE__);       \
      ndll::string error = "[" + file + ":" + line +      \
        "]: NVML error \"" +                              \
        nvmlErrorString(status) + "\"";                   \
      throw std::runtime_error(error);                    \
    }                                                     \
  } while (0)

// For calling NDLL library functions
#define NDLL_CALL(code)                                         \
  do {                                                          \
    NDLLError_t status = code;                                  \
    if (status != NDLLSuccess) {                                \
      ndll::string file = __FILE__;                             \
      ndll::string line = std::to_string(__LINE__);             \
      ndll::string error = NDLLGetLastError();                  \
      throw std::runtime_error(error);                          \
    }                                                           \
  } while (0)

// Excpetion throwing checks for pipeline code
#define ENFRC_1(code)                                                         \
  do {                                                                        \
    if (!(code)) {                                                            \
      ndll::string error = ndll::BuildErrorString(#code, __FILE__, __LINE__); \
      throw std::runtime_error(error);                                        \
    }                                                                         \
  } while (0)

#define ENFRC_2(code, str)                                                    \
  do {                                                                        \
    if (!(code)) {                                                            \
      ndll::string error = ndll::BuildErrorString(#code, __FILE__, __LINE__); \
      ndll::string usr_str = str;                                             \
      error += ": " + usr_str;                                                \
      throw std::runtime_error(error);                                        \
    }                                                                         \
  } while (0)

#define NDLL_ENFORCE(...) GET_MACRO(__VA_ARGS__, ENFRC_2, ENFRC_1)(__VA_ARGS__)

// Enforces that the value of 'var' is in the range [lower, upper)
#define NDLL_ENFORCE_IN_RANGE(var, lower, upper)                                \
  do {                                                                          \
    if (((var) < (lower)) || ((var) >= (upper))) {                              \
      ndll::string error = "Index " + std::to_string(var) + " out of range [" + \
        std::to_string(lower) + ", " + std::to_string(upper) + ").";            \
      NDLL_FAIL(error);                                                         \
    }                                                                           \
  } while (0)

// Enforces that the input var is in the range [0, upper)
#define NDLL_ENFORCE_VALID_INDEX(var, upper) \
  NDLL_ENFORCE_IN_RANGE(var, static_cast<int>(0), upper)

#if NDLL_USE_STACKTRACE && NDLL_DEBUG
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

inline ndll::string GetStacktrace() {
  const int MAX_STACK_SIZE = 100;
  void * stack[MAX_STACK_SIZE];
  int nframes = backtrace(stack, MAX_STACK_SIZE);
  ndll::string ret = "\nStacktrace (" + std::to_string(nframes) + " entries):\n";
  char **msgs = backtrace_symbols(stack, nframes);
  if (msgs != nullptr) {
    for (int frame = 0; frame < nframes; ++frame) {
      ndll::string msg(msgs[frame]);
      size_t symbol_start = string::npos;
      size_t symbol_end = string::npos;
      ndll::string s = msgs[frame];
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
inline ndll::string GetStacktrace() {
  return "";
}
#endif  // NDLL_USE_STACKTRACE && NDLL_DEBUG

#define NDLL_FAIL(str)                                              \
  do {                                                              \
    ndll::string file = __FILE__;                                   \
    ndll::string line = std::to_string(__LINE__);                   \
    ndll::string error_str = "[" + file + ":" + line + "] " + str;  \
    error_str += ndll::GetStacktrace();                                        \
    throw std::runtime_error(error_str);                            \
  } while (0)

void NDLLReportFatalProblem(const char *file, int line, const char *pComment);
#define REPORT_FATAL_PROBLEM(comment) NDLLReportFatalProblem(__FILE__, __LINE__, comment)

}  // namespace ndll

#endif  // NDLL_ERROR_HANDLING_H_
