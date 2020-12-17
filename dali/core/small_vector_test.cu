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
#include "dali/test/device_test.h"
#include "dali/core/small_vector.h"

DEVICE_TEST(SmallVectorDev, Test, dim3(1), dim3(1)) {
  dali::SmallVector<int, 3> v;
  DEV_EXPECT_EQ(v.capacity(), 3u);
  DEV_EXPECT_EQ(v.size(), 0u);
  v.push_back(1);
  v.push_back(3);
  v.push_back(5);
  DEV_EXPECT_FALSE(v.is_dynamic());
  v.push_back(7);
  DEV_EXPECT_TRUE(v.is_dynamic());
  v.insert(v.begin() + 1, 2);
  v.insert(v.begin() + 3, 4);
  v.insert(v.begin() + 5, 6);
  v.insert(v.begin() + 7, 8);
  DEV_EXPECT_EQ(v[0], 1);
  DEV_EXPECT_EQ(v[1], 2);
  DEV_EXPECT_EQ(v[2], 3);
  DEV_EXPECT_EQ(v[3], 4);
  DEV_EXPECT_EQ(v[4], 5);
  DEV_EXPECT_EQ(v[5], 6);
  DEV_EXPECT_EQ(v[6], 7);
  DEV_EXPECT_EQ(v[7], 8);
  v.erase(v.begin()+2, v.end()-2);
  DEV_ASSERT_EQ(v.size(), 4u);
  DEV_EXPECT_EQ(v[0], 1);
  DEV_EXPECT_EQ(v[1], 2);
  DEV_EXPECT_EQ(v[2], 7);
  DEV_EXPECT_EQ(v[3], 8);
}

DEVICE_TEST(SmallVectorDev, MovePoD, 1, 1) {
  dali::SmallVector<int, 4> a, b;
  a.push_back(1);
  a.push_back(2);
  b.push_back(3);
  b = cuda_move(a);
  DEV_EXPECT_EQ(b[0], 1);
  DEV_EXPECT_EQ(b[1], 2);
  DEV_EXPECT_TRUE(a.empty());
  b.push_back(3);
  b.push_back(4);
  b.push_back(5);
  DEV_EXPECT_TRUE(b.is_dynamic());
  auto *ptr = b.data();
  a = cuda_move(b);
  DEV_EXPECT_EQ(a.data(), ptr);
  DEV_EXPECT_TRUE(b.empty());
}


DEVICE_TEST(SmallVectorDev, Resize, 1, 1) {
  dali::SmallVector<int32_t, 4> v;
  v.resize(3, 5);
  DEV_ASSERT_EQ(v.size(), 3u);
  DEV_EXPECT_EQ(v[0], 5);
  DEV_EXPECT_EQ(v[1], 5);
  DEV_EXPECT_EQ(v[2], 5);
  v.resize(16, 42);
  DEV_ASSERT_EQ(v.size(), 16u);
  for (int i = 3; i < 16; i++)
    DEV_EXPECT_EQ(v[i], 42);
  v.resize(6);
  DEV_EXPECT_EQ(v.size(), 6u);
}

// NOTE: this test should compile without warnings
TEST(SmallVector, TestNoExecCheck) {
  struct X {
    __host__ X(int x) : x(x) {}
    __host__ X(const X &x) : x(x.x) {}
    int x;
  };

  dali::SmallVector<X, 4> v1, v2;
  v1.emplace_back(42);  // the constructor
  v2 = v1;
  EXPECT_EQ(v1[0].x, 42);
  EXPECT_EQ(v2[0].x, 42);
}
