// Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/device_guard.h"
#include "dali/core/cuda_error.h"
#include "dali/core/unique_handle.h"

namespace dali {

TEST(DeviceGuard, ConstructorWithDevice) {
  int test_device = 0;
  int guard_device = 0;
  int current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CUDA_CALL(cudaSetDevice(test_device));
  CUDA_CALL(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g(guard_device);
    CUDA_CALL(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
  }
  CUDA_CALL(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
}

TEST(DeviceGuard, ConstructorWithDeviceRestoresNullContextMultiGPU) {
  int test_device = 0;
  int guard_device = 1;
  int current_device = 0;
  int count = 0;
  CUcontext current_context = NULL;

  ASSERT_TRUE(cuInitChecked());
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "Test requires at least two CUDA devices";
  }

  // Restore the state that was current when this test entered.
  DeviceGuard restore_guard;
  CUDA_CALL(cudaSetDevice(test_device));
  CUDA_CALL(cuCtxSetCurrent(NULL));
  CUDA_CALL(cudaGetDevice(&current_device));
  ASSERT_EQ(current_device, test_device);
  CUDA_CALL(cuCtxGetCurrent(&current_context));
  ASSERT_EQ(current_context, nullptr);

  {
    DeviceGuard g(guard_device);
    CUDA_CALL(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);

    void *ptr = nullptr;
    CUDA_CALL(cudaMalloc(&ptr, 1));
    CUDA_CALL(cudaFree(ptr));
    CUDA_CALL(cuCtxGetCurrent(&current_context));
    EXPECT_NE(current_context, nullptr);
  }

  // If there were no primary context at construction, DeviceGuard will create a new one
  CUDA_CALL(cuCtxGetCurrent(&current_context));
  EXPECT_NE(current_context, nullptr);
  CUDA_CALL(cudaGetDevice(&current_device));
  EXPECT_NE(current_device, test_device);
}

TEST(DeviceGuard, ConstructorNoArgs) {
  int test_device = 0;
  int guard_device = 0;
  int current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CUDA_CALL(cudaSetDevice(test_device));
  CUDA_CALL(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g;
    CUDA_CALL(cudaSetDevice(guard_device));
    CUDA_CALL(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
  }
  CUDA_CALL(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
}

namespace {
struct CUDAContext : UniqueHandle<CUcontext, CUDAContext> {
  DALI_INHERIT_UNIQUE_HANDLE(CUcontext, CUDAContext);
  static CUDAContext Create(int flags, CUdevice dev) {
    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CUctxCreateParams params = {};
    CUDA_CALL(cuCtxCreate(&ctx, &params, 0, dev));
#else
    CUDA_CALL(cuCtxCreate(&ctx, 0, dev));
#endif
    return CUDAContext(ctx);
  }

  static void DestroyHandle(CUcontext ctx) {
    CUDA_DTOR_CALL(cuCtxDestroy(ctx));
  }
};

}  // namespace

TEST(DeviceGuard, ConstructorNoArgsRestoresNullContextMultiGPU) {
  int test_device = 0;
  CUdevice cu_guard_device = 0;
  int guard_device = 1;
  int current_device = 0;
  int count = 0;
  CUcontext current_context = NULL;

  ASSERT_TRUE(cuInitChecked());
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "Test requires at least two CUDA devices";
  }

  // Restore the state that was current when this test entered.
  DeviceGuard restore_guard;
  CUDA_CALL(cudaSetDevice(test_device));
  CUDA_CALL(cuCtxSetCurrent(NULL));
  CUDA_CALL(cudaGetDevice(&current_device));
  ASSERT_EQ(current_device, test_device);
  CUDA_CALL(cuCtxGetCurrent(&current_context));
  ASSERT_EQ(current_context, nullptr);

  CUDA_CALL(cuDeviceGet(&cu_guard_device, guard_device));
  CUDAContext guard_context;
  {
    DeviceGuard g;
    guard_context = CUDAContext::Create(0, cu_guard_device);
    CUDA_CALL(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
    CUDA_CALL(cuCtxGetCurrent(&current_context));
    EXPECT_EQ(current_context, guard_context);
  }

  // If there were no primary context at construction, DeviceGuard will create a new one
  CUDA_CALL(cuCtxGetCurrent(&current_context));
  EXPECT_NE(current_context, nullptr);
  CUDA_CALL(cudaGetDevice(&current_device));
  EXPECT_NE(current_device, test_device);
}

TEST(DeviceGuard, NegativeDeviceIsNoOp) {
  CUdevice cu_test_device = 0;
  CUdevice cu_current_device = 0;
  CUcontext current_context = nullptr;

  ASSERT_TRUE(cuInitChecked());

  // Restore the state that was current when this test entered.
  DeviceGuard restore_guard;
  CUDA_CALL(cuDeviceGet(&cu_test_device, 0));
  auto test_context = CUDAContext::Create(0, cu_test_device);
  CUDA_CALL(cuCtxSetCurrent(test_context));
  CUDA_CALL(cuCtxGetCurrent(&current_context));
  ASSERT_EQ(current_context, test_context);

  {
    DeviceGuard g(-1);
  }

  CUDA_CALL(cuCtxGetCurrent(&current_context));
  EXPECT_EQ(current_context, test_context);
  CUDA_CALL(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_device, cu_test_device);
}

TEST(DeviceGuard, CheckcontextMultiGPU) {
  int test_device = 0;
  CUdevice cu_test_device = 0;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device = 0;
  CUcontext cu_current_ctx = nullptr;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CUDA_CALL(cuDeviceGet(&cu_test_device, test_device));
  auto cu_test_ctx = CUDAContext::Create(0, cu_test_device);
  CUDA_CALL(cuCtxSetCurrent(cu_test_ctx));
  CUDA_CALL(cuCtxGetCurrent(&cu_current_ctx));
  CUDA_CALL(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g(guard_device);
    CUDA_CALL(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
    CUDA_CALL(cuCtxGetCurrent(&cu_current_ctx));
    EXPECT_NE(cu_current_ctx, cu_test_ctx);
  }
  CUDA_CALL(cuCtxGetCurrent(&cu_current_ctx));
  CUDA_CALL(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
}

TEST(DeviceGuard, CheckcontextNoArgsMultiGPU) {
  int test_device = 0;
  CUdevice cu_test_device = 0;
  CUcontext cu_current_ctx = nullptr;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  CUDA_CALL(cuDeviceGet(&cu_test_device, test_device));
  auto cu_test_ctx = CUDAContext::Create(0, cu_test_device);

  CUDA_CALL(cuCtxSetCurrent(cu_test_ctx));
  CUDA_CALL(cuCtxGetCurrent(&cu_current_ctx));
  CUDA_CALL(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g;
    CUDA_CALL(cudaSetDevice(guard_device));
    CUDA_CALL(cudaGetDevice(&current_device));
    CUDA_CALL(cuCtxGetCurrent(&cu_current_ctx));
    EXPECT_NE(cu_current_ctx, cu_test_ctx);
  }
  CUDA_CALL(cuCtxGetCurrent(&cu_current_ctx));
  CUDA_CALL(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx);
  EXPECT_EQ(cu_current_device, cu_test_device);
}

}  // namespace dali
