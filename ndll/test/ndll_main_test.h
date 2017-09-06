#ifndef NDLL_TEST_NDLL_MAIN_TEST_H_
#define NDLL_TEST_NDLL_MAIN_TEST_H_

#include "ndll/common.h"
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
    if (status != cudaSuccess) {                    \
      string file = __FILE__;                       \
      string line = std::to_string(__LINE__);       \
      string error = "[" + file + ":" + line +      \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(status) + "\"";          \
      ASSERT_TRUE(false) << error;                  \
    }                                               \
  } while (0)

} // namespace ndll

#endif // NDLL_TEST_NDLL_MAIN_TEST_H_
