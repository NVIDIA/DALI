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
#include "dali/core/call_at_exit.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/device_guard.h"
#include "dali/core/cuda_error.h"
#include "dali/core/unique_handle.h"
#include "dali/test/timing.h"

namespace dali {

int DoTestUninit0() {
  if (!cuInitChecked()) {
    std::cout << "Cannot initialize CUDA." << std::endl;
    return 0xdead;
  }

  unsigned flags = 0;
  int active = 0;
  CUDA_CALL(cuDevicePrimaryCtxGetState(0, &flags, &active));
  if (active) {
    std::cout << "This test cannot be run with a primary context already active for device 0"
              << std::endl;
    return 0xbad;
  }

  int curr = -1;
  {
    DeviceGuard dg(1);
    CUDA_CALL(cudaGetDevice(&curr));
    if (curr != 1) {
      std::cout << "Device not switched properly." << std::endl;
      return 1;
    }
  }
  CUDA_CALL(cudaGetDevice(&curr));
  if (curr != 0) {
    std::cout << "Device not restored properly." << std::endl;
    return 2;
  }
  return 0;
}

void TestUninit0() {
  _exit(DoTestUninit0());
}

int DoTestPrimaryContextInitMultithreaded() {
  if (!cuInitChecked()) {
    std::cout << "Cannot initialize CUDA." << std::endl;
    return 0xdead;
  }

  unsigned flags = 0;
  int active = 0;
  CUDA_CALL(cuDevicePrimaryCtxGetState(0, &flags, &active));
  if (active) {
    std::cout << "This test cannot be run with a primary context already active for device 0";
    return 0xbad;
  }

  std::atomic<bool> start{false};
  bool failure = false;

  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&]() {
      while (!start.load()) {}
      DeviceGuard dg(0);
      unsigned flags = 0;
      int active = 0;
      if (cuDevicePrimaryCtxGetState(0, &flags, &active) != CUDA_SUCCESS) {
        std::cout << "Cannot get primary context for device 0." << std::endl;
        failure = true;
      }
      if (!active) {
        std::cout << "Primary context not active" << std::endl;
        failure = true;
      }
    });
  }

  start.store(true);
  for (auto &t : threads)
    t.join();

  if (failure) {
    std::cout << "A failure was detected in one of the worker threads." << std::endl;
    return 1;
  }

  {
    unsigned flags = 0;
    int active = 0;
    if (cuDevicePrimaryCtxGetState(0, &flags, &active) != CUDA_SUCCESS) {
      std::cout << "Cannot get primary context for device 0." << std::endl;
      return 2;
    }

    if (!active) {
      std::cout << "Primary context should be active as a side-effect" << std::endl;
      return 3;
    }
  }

  return 0;
}

void TestPrimaryContextInitMultithreaded() {
  _exit(DoTestPrimaryContextInitMultithreaded());
}

TEST(DeviceGuard, RestoreUninit0_MultiGPU) {
  int count = 0;
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(TestUninit0(), testing::ExitedWithCode(0), "");
}

TEST(DeviceGuard, Multithreaded) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(TestPrimaryContextInitMultithreaded(), testing::ExitedWithCode(0), "");
}

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

TEST(DeviceGuard, CheckContext) {
  ASSERT_TRUE(cuInitChecked());

  CUcontext old = nullptr;
  CUDA_CALL(cuCtxGetCurrent(&old));
  auto restore = AtScopeExit([old]() {
    CUDA_DTOR_CALL(cuCtxSetCurrent(old));
  });
  auto cu_test_ctx0 = CUDAContext::Create(0, 0);
  auto cu_test_ctx1 = CUDAContext::Create(0, 0);
  CUDA_CALL(cuCtxSetCurrent(cu_test_ctx0));
  CUcontext ctx = nullptr;
  {
    DeviceGuard g;
    CUDA_CALL(cuCtxGetCurrent(&ctx));
    EXPECT_EQ(ctx, cu_test_ctx0.get());
    CUDA_CALL(cuCtxSetCurrent(cu_test_ctx1));
  }
  CUDA_CALL(cuCtxGetCurrent(&ctx));
  EXPECT_EQ(ctx, cu_test_ctx0.get()) << "Context not restored upon construction";
}


TEST(DeviceGuard, CheckContextNoArgs) {
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

TEST(DeviceGuard, NegativeDevicNoOp) {
  // This test creates a DeviceGuard with a negative device ID and destroys it out-of-order
  // to verify that it's really no-op, even with valid CUDA context present.

  ASSERT_TRUE(cuInitChecked());
  int count = 0;
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  int dev = -1;
  {
    DeviceGuard dg0(0);
    std::optional<DeviceGuard> dgneg;
    {
      DeviceGuard dg1(1);
      dgneg.emplace(-1);  // this will outlive the scope
      CUDA_CALL(cudaGetDevice(&dev));
      EXPECT_EQ(dev, 1);
    }
    CUDA_CALL(cudaGetDevice(&dev));
    EXPECT_EQ(dev, 0);
    dgneg.reset();  // shouldn't change anything
    CUDA_CALL(cudaGetDevice(&dev));
    EXPECT_EQ(dev, 0);
  }
  CUDA_CALL(cudaGetDevice(&dev));
  EXPECT_EQ(dev, 0);
}

TEST(DeviceGuard, CheckOutOfRange) {
  int ndevs = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndevs));
  EXPECT_THROW(DeviceGuard{ndevs}, std::out_of_range);
}

TEST(DeviceGuard, SwitchPerf_MultiGPU) {
  ASSERT_TRUE(cuInitChecked());
  int count = 0;
  CUDA_CALL(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  {
    DeviceGuard dg0(0);
    DeviceGuard dg1(1);
  }

  auto start = test::perf_timer::now();
  int iters = 100000;
  for (int i = 0; i < iters; i++) {
    DeviceGuard dg0(0);
    DeviceGuard dg1(1);
  }
  auto end = test::perf_timer::now();
  std::cout << test::format_time(test::seconds(end - start) / (2 * iters)) << std::endl;
}

TEST(DeviceGuard, SameDevPerf) {
  ASSERT_TRUE(cuInitChecked());
  int count = 0;
  DeviceGuard dg0(0);

  auto start = test::perf_timer::now();
  int iters = 100000;
  for (int i = 0; i < iters; i++) {
    DeviceGuard dg0(0);
  }
  auto end = test::perf_timer::now();
  std::cout << test::format_time(test::seconds(end - start) / iters) << std::endl;
}

}  // namespace dali
