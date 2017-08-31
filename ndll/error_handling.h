#ifndef NDLL_ERROR_HANDLING_H_
#define NDLL_ERROR_HANDLING_H_

#include <sstream>
#include <string>

#include <cuda_runtime_api.h>

namespace ndll {

// Note: We won't control all of the code that is executing in this
// library because users can define their own ops. Because of this,
// we can't use fixed error types like other Nvidia libraries. To
// deal with this, we'll define an exception type and allow users
// to pass in error strings so that they can be propagated out to
// the caller when an error occurs.

/**
 * @brief Basic exception class used by ndll error checking.
 */
class NdllException {
public:
  NdllException() {}
  
  explicit NdllException(std::string str) {
    str_.append(str);
  }

  const char* what() {
    return str_.c_str();
  }
  
private:
  std::string str_;
};

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
      throw NdllException(error);                   \
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
      throw NdllException(error);                               \
    }                                                           \
  } while(0)

#define ASRT_2(code, str)                                       \
  do {                                                          \
    if (!code) {                                                \
      string error = GetErrorString(#code, __FILE__, __LINE__); \
      string usr_str = str;                                     \
      error += ": " + usr_str;                                  \
      throw NdllException(error);                               \
    }                                                           \
  } while(0)

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define NDLL_ASSERT(...) GET_MACRO(__VA_ARGS__, ASRT_2, ASRT_1)(__VA_ARGS__)

// Note: We can define error checking for other libraries
// here. E.g. CUBLAS_CALL, NPP_CALL, etc.

} // namespace ndll

#endif // NDLL_ERROR_HANDLING_H_
