#ifndef NDLL_TEST_NDLL_MAIN_TEST_H_
#define NDLL_TEST_NDLL_MAIN_TEST_H_

#include <cassert>
#include <fstream>

#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/util/image.h"

namespace ndll {

// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/image/testing_jpegs";

// Main testing fixture to provide common functionality across tests
class NDLLTest : public ::testing::Test {
public:
  virtual void SetUp() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  }

  virtual void TearDown() {
    for (auto &ptr : jpegs_) delete[] ptr;
  }
  
  int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }
  
protected:
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;
}; 
} // namespace ndll

#endif // NDLL_TEST_NDLL_MAIN_TEST_H_
