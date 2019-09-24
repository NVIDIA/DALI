// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_DEVICE_TEST_H_
#define DALI_TEST_DEVICE_TEST_H_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "dali/core/cuda_utils.h"
#include "dali/core/dev_string.h"

#define MAX_DEVICE_ERROR_MESSAGES 100

namespace dali {
namespace testing {

struct TestStatus {
  bool failed;
  bool fatal;
  int num_messages;
};

}  // namespace testing
}  // namespace dali


#define DEV_CHECK_OP_AB(a, b, op, is_fatal) { \
  decltype(a) a_result = a; \
  decltype(b) b_result = b; \
  if (!(a_result op b_result)) { \
    auto a_str = dali::dev_to_string(a_result); \
    auto b_str = dali::dev_to_string(b_result); \
    int nmsg = atomicAdd(&test_status.num_messages, 1); \
    if (nmsg < MAX_DEVICE_ERROR_MESSAGES && !test_status.fatal) \
      printf("Check failed at %s:%i:\n%s ( = %s) " #op " %s ( = %s)\n", \
          __FILE__, __LINE__, \
          #a, a_str.c_str(), #b, b_str.c_str()); \
    test_status.failed = true; \
    if (is_fatal) test_status.fatal = true; \
  } \
}

#define DEV_BOOLEAN_CHECK_OP(cond, op, bad_result, is_fatal) { \
  if (op(cond)) { \
    int nmsg = atomicAdd(&test_status.num_messages, 1); \
    if (nmsg < MAX_DEVICE_ERROR_MESSAGES && !test_status.fatal) \
      printf("Check failed at %s:%i:\n%s evaluates to " #bad_result "\n", \
          __FILE__, __LINE__, #cond); \
    test_status.failed = true; \
    if (is_fatal) test_status.fatal = true; \
  } \
}

#define DEV_CHECK_TRUE(cond, is_fatal) DEV_BOOLEAN_CHECK_OP(cond, !, false, is_fatal)
#define DEV_CHECK_FALSE(cond, is_fatal) DEV_BOOLEAN_CHECK_OP(cond, (bool), true, is_fatal)

#define DEV_EXPECT_EQ(a, b)\
  DEV_CHECK_OP_AB(a, b, ==, false)

#define DEV_EXPECT_NE(a, b)\
  DEV_CHECK_OP_AB(a, b, !=, false)

#define DEV_EXPECT_LT(a, b)\
  DEV_CHECK_OP_AB(a, b, <, false)

#define DEV_EXPECT_GT(a, b)\
  DEV_CHECK_OP_AB(a, b, >, false)

#define DEV_EXPECT_LE(a, b)\
  DEV_CHECK_OP_AB(a, b, <=, false)

#define DEV_EXPECT_GE(a, b)\
  DEV_CHECK_OP_AB(a, b, >=, false)

#define DEV_EXPECT_TRUE(cond)\
  DEV_CHECK_TRUE(cond, false)

#define DEV_EXPECT_FALSE(cond)\
  DEV_CHECK_FALSE(cond, false)

#define DEV_ASSERT_EQ(a, b)\
  DEV_CHECK_OP_AB(a, b, ==, true)

#define DEV_ASSERT_NE(a, b)\
  DEV_CHECK_OP_AB(a, b, !=, true)

#define DEV_ASSERT_LT(a, b)\
  DEV_CHECK_OP_AB(a, b, <, true)

#define DEV_ASSERT_GT(a, b)\
  DEV_CHECK_OP_AB(a, b, >, true)

#define DEV_ASSERT_LE(a, b)\
  DEV_CHECK_OP_AB(a, b, <=, true)

#define DEV_ASSERT_GE(a, b)\
  DEV_CHECK_OP_AB(a, b, >=, true)

#define DEV_ASSERT_TRUE(cond)\
  DEV_CHECK_TRUE(cond, true)

#define DEV_ASSERT_FALSE(cond)\
  DEV_CHECK_FALSE(cond, true)

#define DEV_FAIL(...) { test_status->failed = true; test_status->fatal = true; return __VA_ARGS__; }

#define DECLARE_TEST_KERNEL(suite_name, test_name)\
template <typename... Args>\
__global__ void suite_name##_##test_name##_kernel( \
    dali::testing::TestStatus *test_status, Args... args)

#define DEFINE_TEST_KERNEL(suite_name, test_name, ...) \
__device__ void suite_name##_##test_name##_body( \
    dali::testing::TestStatus &test_status, ##__VA_ARGS__); \
\
template <typename... Args>\
__global__ void suite_name##_##test_name##_kernel(\
      dali::testing::TestStatus *test_status, Args... args) { \
  if (!test_status->fatal) {\
    __syncthreads(); \
    suite_name##_##test_name##_body(*test_status, args...); \
  } \
} \
__device__ void suite_name##_##test_name##_body( \
    dali::testing::TestStatus &test_status, ##__VA_ARGS__)

#define TEST_KERNEL_NAME(suite_name, test_name) suite_name##_##test_name##_kernel

/**
 * Executes default test case body.
 * @param suite_name - test suite name, as used in DEFINE_TEST_KERNEL
 * @param test_name - test case name, as used in DEFINE_TEST_KERNEL
 * @param grid - CUDA grid size
 * @param block - CUDA block size
 * @param ... - extra parameters passed to the kernel invocation, if any
 */
#define DEVICE_TEST_CASE_BODY(suite_name, test_name, grid, block, ...) \
  using TestStatus = dali::testing::TestStatus; \
  TestStatus *status = nullptr; \
  cudaMalloc(reinterpret_cast<void**>(&status), sizeof(TestStatus)); \
  ASSERT_NE(status, nullptr) << "Cannot allocate test status block"; \
  cudaMemset(status, 0, sizeof(TestStatus)); \
  cudaGetLastError(); \
  suite_name##_##test_name##_kernel<<<grid, block>>>(status, ##__VA_ARGS__); \
  dali::testing::TestStatus host_status = {0}; \
  cudaMemcpy(&host_status, status, sizeof(TestStatus), cudaMemcpyDeviceToHost); \
  cudaFree(status); \
  auto err = cudaGetLastError(); \
  EXPECT_EQ(err, cudaSuccess) << "CUDA error: " \
    << cudaGetErrorName(err) << " " << cudaGetErrorString(err); \
  if (err == cudaErrorIllegalAddress || err == cudaErrorIllegalInstruction) { \
    std::cerr << "A fatal CUDA error was reported. Resetting the device!" << std::endl; \
    exit(err); \
  } \
  EXPECT_FALSE(host_status.failed) << "There were errors in device code";

/**
 * Simple test of a device function
 * @param suite_name GTest's suite name
 * @param test_name GTest's test case name
 * @param grid CUDA grid size
 * @param block CUDA block size
 */
#define DEVICE_TEST(suite_name, test_name, grid, block) \
DECLARE_TEST_KERNEL(suite_name, test_name); \
TEST(suite_name, test_name) \
{ DEVICE_TEST_CASE_BODY(suite_name, test_name, grid, block); } \
DEFINE_TEST_KERNEL(suite_name, test_name)

#endif  // DALI_TEST_DEVICE_TEST_H_
