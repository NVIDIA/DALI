#include <iostream>

#include <gtest/gtest.h>

#include <ndll/error_handling.h>

namespace ndll {

TEST(DummyTests, DummyTest) {
  // Do some dumb stuff
  try {
    if (true)
      NDLL_ASSERT(false) << "user error stuff";
  } catch(ndll::NdllException &e) {
    std::cout << e.what() << std::endl;
  }

  try {
  // CUDA_CALL(cudaMalloc(nullptr, 5));
  } catch(ndll::NdllException &e) {
    std::cout << e.what() << std::endl;
  }
}

} // namespace ndll
