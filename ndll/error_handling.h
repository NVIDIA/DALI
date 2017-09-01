#ifndef NDLL_ERROR_HANDLING_H_
#define NDLL_ERROR_HANDLING_H_

#include <sstream>
#include <string>

#include <cuda_runtime_api.h>

namespace ndll {

/**
 * @brief Basic exception class used by ndll error checking.
 */ 
class NDLLException {
public:
  NDLLException() {}
  
  explicit NDLLException(std::string str) {
    str_.append(str);
  }

  const char* what() {
    return str_.c_str();
  }
  
private:
  string str_;
};

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

// Note: If the library ever gets more complicated than data
// augmentation primitives we will need more sophisticated
// error handling. Errors will need to be propagated out of
// nested functions and passed back to the user.

// CUDA error checking utility
#define CUDA_CALL(code)                             \
  do {                                              \
    cudaError_t status = code;                      \
    if (code != cudaSuccess) {                      \
      std::string file = __FILE__;                  \
      std::string line = std::to_string(__LINE__);  \
      std::string error = "[" + file + ":" + line + \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(code) + "\"";            \
      return NDLLCudaError;                    \
    }                                               \
  } while (0)

inline string GetErrorString(string statement, string file, int line) {
  string line_str = std::to_string(line);
  std::string error = "[" + file + ":" + line_str +
    "]: Assert on \"" + statement +
    "\" failed";
  return error;
}

#define ASRT_1(code)                                            \
  do {                                                          \
    if (!code) {                                                \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      return NDLLError;                                         \
    }                                                           \
  } while (0)

#define ASRT_2(code, str)                                       \
  do {                                                          \
    if (!code) {                                                \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      string usr_str = str;                                     \
      error += ": " + usr_str;                                  \
      return NDLLError;                                         \
    }                                                           \
  } while (0)

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define NDLL_ASSERT(...) GET_MACRO(__VA_ARGS__, ASRT_2, ASRT_1)(__VA_ARGS__)

// Note: We can define error checking for other libraries
// here. E.g. CUBLAS_CALL, NPP_CALL, etc.

} // namespace ndll

#endif // NDLL_ERROR_HANDLING_H_
