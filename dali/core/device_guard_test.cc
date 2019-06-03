// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/device_guard.h"

namespace dali {

TEST(DeviceGuard, ConstructorWithDevice) {
  int test_device = 0;
  int guard_device = 0;
  int current_device;
  int count = 1;

  EXPECT_EQ(cuInitChecked(), true);
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  EXPECT_EQ(cudaSetDevice(test_device), cudaSuccess);
  EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g(guard_device);
    EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, guard_device);
  }
  EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, test_device);
}

TEST(DeviceGuard, ConstructorNoArgs) {
  int test_device = 0;
  int guard_device = 0;
  int current_device;
  int count = 1;

  EXPECT_EQ(cuInitChecked(), true);
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  EXPECT_EQ(cudaSetDevice(test_device), cudaSuccess);
  EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g;
    EXPECT_EQ(cudaSetDevice(guard_device), cudaSuccess);
    EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, guard_device);
  }
  EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, test_device);
}

TEST(DeviceGuard, Checkcontext) {
  int test_device = 0;
  CUdevice cu_test_device;
  CUcontext cu_test_ctx;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device;
  CUcontext cu_current_ctx;
  int count = 1;

  EXPECT_EQ(cuInitChecked(), true);
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  EXPECT_EQ(cuDeviceGet(&cu_test_device, test_device), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxCreate(&cu_test_ctx, 0, cu_test_device), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxSetCurrent(cu_test_ctx), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxGetCurrent(&cu_current_ctx), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxGetDevice(&cu_current_device), CUDA_SUCCESS);
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g(guard_device);
    EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, guard_device);
    EXPECT_EQ(cuCtxGetCurrent(&cu_current_ctx), CUDA_SUCCESS);
    EXPECT_NE(cu_current_ctx, cu_test_ctx);
  }
  EXPECT_EQ(cuCtxGetCurrent(&cu_current_ctx), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxGetDevice(&cu_current_device), CUDA_SUCCESS);
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
  cuCtxDestroy(cu_test_ctx);
}

TEST(DeviceGuard, CheckcontextNoArgs) {
  int test_device = 0;
  CUdevice cu_test_device;
  CUcontext cu_test_ctx;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device;
  CUcontext cu_current_ctx;
  int count = 1;

  EXPECT_EQ(cuInitChecked(), true);
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  EXPECT_EQ(cuDeviceGet(&cu_test_device, test_device), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxCreate(&cu_test_ctx, 0, cu_test_device), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxSetCurrent(cu_test_ctx), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxGetCurrent(&cu_current_ctx), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxGetDevice(&cu_current_device), CUDA_SUCCESS);
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g;
    EXPECT_EQ(cudaSetDevice(guard_device), cudaSuccess);
    EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(cuCtxGetCurrent(&cu_current_ctx), CUDA_SUCCESS);
    EXPECT_NE(cu_current_ctx, cu_test_ctx);
  }
  EXPECT_EQ(cuCtxGetCurrent(&cu_current_ctx), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxGetDevice(&cu_current_device), CUDA_SUCCESS);
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
  cuCtxDestroy(cu_test_ctx);
}

}  // namespace dali
