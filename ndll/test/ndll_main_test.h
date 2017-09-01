#ifndef NDLL_TEST_NDLL_MAIN_TEST_H_
#define NDLL_TEST_NDLL_MAIN_TEST_H_

#include "ndll/error_handling.h"

namespace ndll {

// Error checking for the test suite
#define CHECK_NDLL(error)                       \
  do {                                          \
    ASSERT_FALSE(error);                        \
  } while (0)

#define CHECK_CUDA(code)                            \
  do {                                              \
    cudaError_t status = code;                      \
    if (code != cudaSuccess) {                      \
      std::string file = __FILE__;                  \
      std::string line = std::to_string(__LINE__);  \
      std::string error = "[" + file + ":" + line + \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(code) + "\"";            \
      ASSERT_TRUE(false) << error;                  \
    }                                               \
  } while (0)

} // namespace ndll

#endif // NDLL_TEST_NDLL_MAIN_TEST_H_
