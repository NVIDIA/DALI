#ifndef NDLL_ERROR_HANDLING_H_
#define NDLL_ERROR_HANDLING_H_

#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <nvml.h>

#include "ndll/common.h"
#include "ndll/util/npp.h"

namespace ndll {

// TODO(tgale): We need a better way of handling internal
// cuda errors. We should really have some way of getting
// the error string from cuda.

/**
 * @brief Error object returned by ndll functions
 */
enum NDLLError_t {
  NDLLSuccess = 0,
  NDLLError = 1 /* Something bad happened */, 
  NDLLCudaError = 2 /* CUDA broke */
};

/*
// CUDA error checking utility
#define CUDA_CALL(code)                             \
  do {                                              \
    cudaError_t status = code;                      \
    if (status != cudaSuccess) {                    \
      string file = __FILE__;                       \
      string line = std::to_string(__LINE__);       \
      string error = "[" + file + ":" + line +      \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(status) + "\"";          \
      return NDLLCudaError;                         \
    }                                               \
  } while (0)
*/

inline string GetErrorString(string statement, string file, int line) {
  string line_str = std::to_string(line);
  string error = "[" + file + ":" + line_str +
    "]: Assert on \"" + statement +
    "\" failed";
  return error;
}

#define ASRT_1(code)                                            \
  do {                                                          \
    if (!(code)) {                                              \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      cout << error << endl;                                    \
      return NDLLError;                                         \
    }                                                           \
  } while (0)

#define ASRT_2(code, str)                                       \
  do {                                                          \
    if (!(code)) {                                              \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      string usr_str = str;                                     \
      error += ": " + usr_str;                                  \
      cout << error << endl;                                    \
      return NDLLError;                                         \
    }                                                           \
  } while (0)

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define NDLL_ASSERT(...) GET_MACRO(__VA_ARGS__, ASRT_2, ASRT_1)(__VA_ARGS__)

#define NDLL_FORWARD_ERROR(code) \
  if ((code) == NDLLError) {     \
    return NDLLError;            \
  }

// For checking npp return errors in ndll library functions
#define NDLL_CHECK_NPP(code)                        \
  do {                                              \
    NppStatus status = code;                        \
    if (status != NPP_SUCCESS) {                    \
      string file = __FILE__;                       \
      string line = std::to_string(__LINE__);       \
      string error = "[" + file + ":" + line +      \
        "]: NPP error \"" +                         \
        nppErrorString(status) + "\"";              \
      cout << error << endl;                        \
      return NDLLError;                             \
    }                                               \
  } while (0)


// For calling CUDA library functions
#define CUDA_CALL(code)                             \
  do {                                              \
    cudaError_t status = code;                      \
    if (status != cudaSuccess) {                    \
      string file = __FILE__;                       \
      string line = std::to_string(__LINE__);       \
      string error = "[" + file + ":" + line +      \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(status) + "\"";          \
      throw std::runtime_error(error);              \
    }                                               \
  } while (0)

// For calling NVML library functions
#define NVML_CALL(code)                             \
  do {                                              \
    nvmlReturn_t status = code;                     \
    if (status != NVML_SUCCESS) {                   \
      string file = __FILE__;                       \
      string line = std::to_string(__LINE__);       \
      string error = "[" + file + ":" + line +      \
        "]: NVML error \"" +                        \
        nvmlErrorString(status) + "\"";             \
      throw std::runtime_error(error);              \
    }                                               \
  } while (0)

// For calling NDLL library functions
#define NDLL_CALL(code)                                         \
  do {                                                          \
    NDLLError_t status = code;                                  \
    if (status != NDLLSuccess) {                                \
      string file = __FILE__;                                   \
      string line = std::to_string(__LINE__);                   \
      string error = "[" + file + ":" + line +                  \
        "]: NDLL error";                                        \
      throw std::runtime_error(error);                          \
    }                                                           \
  } while (0)

// Excpetion throwing checks for pipeline code
#define ENFRC_1(code)                                           \
  do {                                                          \
    if (!(code)) {                                              \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      throw std::runtime_error(error);                          \
    }                                                           \
  } while (0)

#define ENFRC_2(code, str)                                      \
  do {                                                          \
    if (!(code)) {                                              \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      string usr_str = str;                                     \
      error += ": " + usr_str;                                  \
      throw std::runtime_error(error);                          \
    }                                                           \
  } while (0)

#define NDLL_ENFORCE(...) GET_MACRO(__VA_ARGS__, ENFRC_2, ENFRC_1)(__VA_ARGS__)

#define NDLL_FAIL(str)                                    \
  do {                                                    \
    string file = __FILE__;                               \
    string line = std::to_string(__LINE__);               \
    string error = "[" + file + ":" + line + " " + str;   \
    throw std::runtime_error(str);                        \
  } while (0)
  
} // namespace ndll

#endif // NDLL_ERROR_HANDLING_H_
