#include <iostream>

#include <gtest/gtest.h>

#include "ndll/allocator.h"
#include "ndll/batch.h"
#include "ndll/error_handling.h"
#include "ndll/operator.h"
#include "ndll/stream_pool.h"

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
  CUDA_CALL(cudaMalloc(nullptr, 5));
  } catch(ndll::NdllException &e) {
    std::cout << e.what() << std::endl;
  }

  // Allocate mem
  void *ptr = GpuAllocator::New(10);

  Batch<GpuAllocator, float> batch;
}

} // namespace ndll
