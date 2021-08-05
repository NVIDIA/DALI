// Copyright (c) 2018, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/error_handling.h"

namespace dali {

TEST(Error, EnforceFailed) {
  std::string file_and_line;
  std::string message = "Test message";
  try {
    // the two statements below must be in one line!
    file_and_line = FILE_AND_LINE; DALI_ENFORCE(!"Always fail", message);
    FAIL() << "Exception was expeceted";
  } catch (DALIException &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find(file_and_line), std::string::npos)
      << "File/line spec not found in error `what()`, which is:\n" << msg;
    EXPECT_NE(msg.find(message), std::string::npos)
      << "Original message not found in error `what()`, which is:\n" << msg;
  } catch (...) {
    FAIL() << "Expected DALIException, got other exception";
  }
}

TEST(Error, CudaError) {
  try {
    CUDA_CALL(cudaSetDevice(-2));
    FAIL() << "Exception was expeceted";
  } catch (CUDAError &e) {
    EXPECT_TRUE(e.is_rt_api());
    EXPECT_FALSE(e.is_drv_api());
    EXPECT_EQ(e.rt_error(), cudaErrorInvalidDevice);
    EXPECT_NE(strstr(e.what(), "cudaErrorInvalidDevice"), nullptr)
      << "Error name `cudaErrorInvalidDevice` should have appeared in the exception message\n";
  } catch (...) {
    FAIL() << "Expected CUDAError, got other exception";
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess)  << "Last error not cleared!";
}

TEST(Error, CudaAlloc_Drv) {
  ASSERT_TRUE(cuInitChecked());
  char name[64];
  try {
    CUDA_CALL(cuDeviceGetName(name, sizeof(name), -2));
  } catch (CUDAError &e) {
    EXPECT_TRUE(e.is_drv_api());
    EXPECT_FALSE(e.is_rt_api());
    EXPECT_EQ(e.drv_error(), CUDA_ERROR_INVALID_DEVICE);
    EXPECT_NE(strstr(e.what(), "CUDA_ERROR_INVALID_DEVICE"), nullptr)
      << "Error name `cudaErrorInvalidDevice` should have appeared in the exception message\n";
  }
  CUdevice device;
  cuDeviceGet(&device, 0);
  EXPECT_NO_THROW(CUDA_CALL(cuDeviceGetName(name, sizeof(name), device)));
}

TEST(Error, CudaAlloc) {
  void *mem = nullptr;
  size_t sz = 1_uz << 62;
  EXPECT_THROW(CUDA_CALL(cudaMalloc(&mem, sz)), CUDABadAlloc);
}

}  // namespace dali
