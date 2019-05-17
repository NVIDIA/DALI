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

DEVICE_TEST(SmallVector, DeviceTest, dim3(1), dim3(1)) {
  dali::SmallVector<int, 3, dali::device_side_allocator<int>> v;
  DEV_EXPECT_EQ(v.capacity(), 3);
  DEV_EXPECT_EQ(v.size(), 0);
  v.push_back(1);
  v.push_back(3);
  v.push_back(5);
  DEV_EXPECT_FALSE(v.is_dynamic())
  v.push_back(7);
  DEV_EXPECT_TRUE(v.is_dynamic());
}
