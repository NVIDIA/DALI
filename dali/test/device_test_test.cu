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

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "dali/test/device_test.h"

DEVICE_TEST(DeviceTest, DeviceSideSuccess, dim3(5), dim3(32)) {
  DEV_EXPECT_LT(threadIdx.x, blockDim.x);
  DEV_EXPECT_LT(4, 5);
  DEV_EXPECT_GT(5, 4);
  DEV_EXPECT_EQ(2, 2);
  DEV_EXPECT_NE(-1, 1);
  DEV_EXPECT_LE(4, 4);
  DEV_EXPECT_LE(4, 5);
  DEV_EXPECT_GE(-2, -2);
  DEV_EXPECT_GE(-1, -2);
  DEV_EXPECT_TRUE(threadIdx.x == threadIdx.x);
  DEV_EXPECT_FALSE(threadIdx.x != threadIdx.x);
}

DEFINE_TEST_KERNEL(DeviceTest, DeviceSideFailure) {
  DEV_EXPECT_EQ(threadIdx.x, 0u);
  DEV_EXPECT_LT(4, 4);
  DEV_EXPECT_LT(5, 4);

  DEV_EXPECT_GT(2, 2);
  DEV_EXPECT_GT(2, 3);

  DEV_EXPECT_EQ(2, 3);
  DEV_EXPECT_NE(-1, -1);
  DEV_EXPECT_LE(4, 3);
  DEV_EXPECT_GE(-2, -1);
  DEV_EXPECT_FALSE(threadIdx.x == threadIdx.x);
  DEV_EXPECT_TRUE(threadIdx.x != threadIdx.x);
}

TEST(DeviceTest, DeviceSideFailure) {
  int num = 0;
  EXPECT_NONFATAL_FAILURE({
    DEVICE_TEST_CASE_BODY(DeviceTest, DeviceSideFailure, dim3(1), dim3(2))
    num = status.host.num_messages;
  }, "There were errors in device code");
  EXPECT_EQ(num, 21);  // threadIdx.x == 0 succeeds once
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
