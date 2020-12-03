// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_DALI_CUDA_FINALIZE_TEST_H_
#define DALI_TEST_DALI_CUDA_FINALIZE_TEST_H_

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "dali/core/cuda_utils.h"

namespace dali {

/**
 * @brief GoogleTest Event Listener, that checks after every Test Case if there were
 * no CUDA related errors.
 *
 * We don't want to use TearDown methods that would need to be either injected into every
 * Fixture or defined globally and would run once. Events allow us to define additional
 * checks with any granularity we want.
 *
 * We can use GTest assertion macros in the events with exception of OnTestPartResult.
 */
class CudaFinalizeEventListener : public ::testing::EmptyTestEventListener {
  void OnTestEnd(const ::testing::TestInfo& test_info) override {
    // Check if the driver wrapper is initialized preventing CPU-only tests from
    // using CUDA calls.
    if (cuInitChecked()) {
      auto sync_result = cudaDeviceSynchronize();
      EXPECT_EQ(sync_result, cudaSuccess) << "CUDA error: \"" << cudaGetErrorName(sync_result)
                                          << "\" - " << cudaGetErrorString(sync_result);
      auto err = cudaGetLastError();
      EXPECT_EQ(err, cudaSuccess) << "CUDA error: \"" << cudaGetErrorName(err) << "\" - "
                                  << cudaGetErrorString(err);
    }
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_CUDA_FINALIZE_TEST_H_
